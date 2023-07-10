import pandas as pd
import psycopg2


class PostgreSqlConnector(object):
    def __init__(self, db_args: dict):
        self.db_args = db_args

    def send_sql_query(self, query: str) -> None:
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.
        :param args: аргументы для подключения в БД.
        """
        conn = psycopg2.connect(**self.db_args)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print(f'Error while fetching data from PostgreSQL {error}')
        finally:
            if conn:
                cursor.close()
                conn.close()

    def get_df_from_query(self, query: str) -> pd.DataFrame:
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.
        :param args: аргументы для подключения в БД.

        :return df: датафрейм с результатом.
        """
        conn = psycopg2.connect(**self.db_args)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
