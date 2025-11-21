import logging
import math
import os
import re

import pandas as pd
import sqlglot
from numpy import nan
from pyspark.sql.dataframe import DataFrame as spark_df
from pyspark.sql.utils import AnalysisException

from splink.internals.database_api import AcceptableInputTableType, DatabaseAPI
from splink.internals.spark.database_api import SparkAPI
from splink.internals.dialects import (
    SparkDialect,
)
from splink.internals.misc import (
    major_minor_version_greater_equal_than,
)
from pyspark.sql.types import StringType, IntegerType, DoubleType
import jellyfish

logger = logging.getLogger(__name__)


class DatabricksAPI(SparkAPI):
    sql_dialect = SparkDialect()

    def __init__(
        self,
        *,
        spark_session,
        dataframe_break_lineage_method="delta_lake_table",
        splink_uc_catalog='main',
        splink_uc_database='default',
        splink_uc_volume='splink'
    ):
        DatabaseAPI.__init__(self)
        self.in_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
        if not self.in_databricks:
            raise Exception(
                "This is not a databricks environment. Please use the spark API instead."
            )
        self.break_lineage_method = dataframe_break_lineage_method
        self.spark = spark_session
        self._set_splink_datastore(splink_uc_catalog, splink_uc_database,splink_uc_volume)

        self._register_python_udfs()

    def _set_splink_datastore(self, catalog, database, volume):
        
        # set the catalog and database of where to write output tables
        catalog = (
            catalog if catalog is not None else self.spark.catalog.currentCatalog()
        )
        database = (
            database if database is not None else self.spark.catalog.currentDatabase()
        )
        volume = volume if volume is not None else "splink"
        # this defines the catalog.database location where splink's data outputs will
        # be stored. The filter will remove none, so if catalog is not provided

        self.splink_data_store = ".".join(
            [self._quote_if_needed(x) for x in [catalog, database] if x is not None]
        )
        self.splink_data_store_path = "/".join(
            [x for x in [catalog, database, volume] if x is not None]
        )

    def _register_python_udfs(self):
        # TODO: this should check if these are already registered and skip if so
        # to cut down on warnings
        for func, return_type, spark_name in [
            (jellyfish.jaro_winkler_similarity, DoubleType(), 'jaro_winkler'),
            (jellyfish.jaccard_similarity, IntegerType(), 'jaccard_sim'),
            (jellyfish.jaro_similarity, DoubleType(), 'jaro_sim'),
            (jellyfish.damerau_levenshtein_distance, IntegerType(), 'DAMERAU_LEVENSHTEIN'),
        ]:
            def func_with_null_handling(left, right):
                if left is None or right is None:
                    return None
                else:
                    return func(left, right)
                
            _ = self.spark.udf.register(spark_name, func_with_null_handling, return_type)
            logger.debug(f"Registered UDF {spark_name}")
        
    def _break_lineage_and_repartition(self, spark_df, templated_name, physical_name):

        regex_to_persist = [
            r"__splink__df_comparison_vectors",
            r"__splink__df_concat_sample",
            r"__splink__df_concat_with_tf",
            r"__splink__df_predict",
            r"__splink__df_tf_.+",
            r"__splink__df_representatives.*",
            r"__splink__representatives.*",
            r"__splink__df_neighbours",
            r"__splink__df_connected_components_df",
            r"__splink__blocked_id_pairs",
            r"__splink__marginal_exploded_ids_blocking_rule.*",
            r"__splink__nodes_in_play",
            r"__splink__edges_in_play",
            r"__splink__clusters_at_threshold",
            r"__splink__distinct_clusters_at_threshold",
            r"__splink__clusters_at_all_thresholds",
            r"__splink__clustering_output_final",
            r"__splink__stable_nodes_at_new_threshold",
        ]

        if re.fullmatch(r"|".join(regex_to_persist), templated_name):
            
            if self.break_lineage_method == "delta_lake_files":
                write_path = f"/Volumes/{self.splink_data_store_path}/{physical_name}"
                spark_df.write.mode("overwrite").format("delta").save(write_path)
                spark_df = self.spark.read.format("delta").load(write_path)
                logger.debug(f"Wrote {templated_name} to Delta files at {write_path}")

            elif self.break_lineage_method == "delta_lake_table":
                write_path = f"{self.splink_data_store}.{physical_name}"
                spark_df.write.mode("overwrite").saveAsTable(write_path)
                spark_df = self.spark.table(write_path)
                logger.debug(
                    f"Wrote {templated_name} to Delta Table at "
                    f"{write_path}"
                )
            else:
                raise ValueError(
                    f"Unknown break_lineage_method: {self.break_lineage_method}. Allowed values: delta_lake_files or delta_lake_table"
                )

        return spark_df

