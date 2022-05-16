# coding: utf-8
# create by Changjian 11/1/2018

import sqlite3
import logging

# logger = logging.getLogger("database.log")
# logger.setLevel(logging.DEBUG)

conn = sqlite3.connect("./text.db")

print("opened database successfully")
c = conn.cursor()
c.execute('''CREATE TABLE COMPANY
       (ID INT PRIMARY KEY     NOT NULL,
       NAME           TEXT    NOT NULL,
       AGE            INT     NOT NULL,
       ADDRESS        CHAR(50),
       SALARY         REAL);''')
print("Table created successfully")
conn.commit()
conn.close()