import asyncio
from aiosqlite import connect
from asyncio import Queue
from utils import handle_sqlite_lock
from log import log_general

class DatabaseConnectionPool:
    def __init__(self, db_path):
        self.db_path = db_path
        self._pool = []
        self._max_connections = 15
        self.read_queue = asyncio.Queue()
        self.write_queue = asyncio.Queue()
        asyncio.create_task(self._manage_queue(self.read_queue, self.process_read))
        asyncio.create_task(self._manage_queue(self.write_queue, self.process_write))

    async def _create_connection(self):
        return await connect(self.db_path)

    async def _get_connection(self):
        if self._pool:
            return self._pool.pop(0)
        else:
            return await self._create_connection()

    async def _release_connection(self, connection):
        if len(self._pool) < self._max_connections:
            self._pool.append(connection)
        else:
            await connection.close()

    async def _manage_queue(self, queue, process_function):
        while True:
            query_or_statement, params, fut, is_batch = await queue.get()
            connection = await self._get_connection()
            try:
                result = await process_function(query_or_statement, params, connection, is_batch)
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
            finally:
                await self._release_connection(connection)
            queue.task_done()

    async def read(self, query, params=None):
        fut = asyncio.Future()
        await self.read_queue.put((query, params, fut, False))  # Read operations are not batched
        return await fut

    async def write(self, statement, params=None):
        fut = asyncio.Future()
        is_batch = params and isinstance(params, (list, tuple)) and all(isinstance(p, tuple) for p in params)
        await self.write_queue.put((statement, params, fut, is_batch))
        result = await fut
        return result

    @staticmethod
    @handle_sqlite_lock()
    async def process_read(query, params, connection, is_batch=False):
        if params is None:
            async with connection.execute(query) as cursor:
                return await cursor.fetchall()
        else:
            async with connection.execute(query, params) as cursor:
                return await cursor.fetchall()

    @staticmethod
    @handle_sqlite_lock()
    async def process_write(statement, params, connection, is_batch=False):
        if is_batch:
            async with connection.executemany(statement, params) as cursor:
                await connection.commit()
                return cursor.rowcount  # Returns the number of rows affected
        elif params is None:
            async with connection.execute(statement) as cursor:
                await connection.commit()
                return cursor.rowcount  # Returns the number of rows affected for non-batched operations
        else:
            async with connection.execute(statement, params) as cursor:
                await connection.commit()
                return cursor.rowcount

class PoolManager:
    pools = {}

    @staticmethod
    def get_pool(db_path):
        if db_path not in PoolManager.pools:
            PoolManager.pools[db_path] = DatabaseConnectionPool(db_path)
        return PoolManager.pools[db_path]