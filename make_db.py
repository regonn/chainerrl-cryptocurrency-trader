# -*- coding: utf-8 -*-

import calendar
from datetime import datetime as dt
from dateutil import parser
from pandas import DataFrame, to_datetime
import sqlite3 as lite
import ccxt
import pandas.io.sql as psql
import time

# 取得回数
FETCH_COUNT = 100

# 一回で取得する範囲
LIMIT = 500

con = lite.connect('btcusd.db', isolation_level=None,
                   detect_types=lite.PARSE_DECLTYPES | lite.PARSE_COLNAMES)
cur = con.cursor()

bitmex = ccxt.bitmex()

symbol = 'BTC/USD'

cols = ['date', 'open', 'high', 'low', 'close', 'volume']


def fetch_ohlcv(since):
    ohlcv = bitmex.fetch_ohlcv(symbol, timeframe='1m',
                               since=since, limit=LIMIT, params={'partial': False})

    df = DataFrame(ohlcv, columns=cols)
    df['date'] = to_datetime(df['date'], unit='ms',
                             utc=True, infer_datetime_format=True)
    print(df)
    df.to_sql('ticks', con, if_exists='append', index=None)
    print("fetch done")


# CREATE DB
cur.execute(
    "CREATE TABLE ticks(date datetime unique, open real, high real, low real, close real, volume integer)")

# RESET DB
cur.execute(
    "delete from ticks"
)

now = dt.now()
unix = int(now.timestamp())
since = (unix - 60 * LIMIT) * 1000

fetch_ohlcv(since)

for i in range(FETCH_COUNT - 1):
    # 高頻度で取得すると、APIが弾かれるので時間をあける
    time.sleep(2)
    cur.execute('SELECT min(date) from ticks')
    sql = cur.fetchall()
    parsed = parser.parse(sql[0][0])
    unixtime = calendar.timegm(parsed.utctimetuple())
    target_since = (unixtime - 60 * LIMIT) * 1000

    fetch_ohlcv(target_since)