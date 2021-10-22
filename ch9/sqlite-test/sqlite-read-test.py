import sqlite3
import os
import numpy as np

# Create a connection to SQLite database (it creates a database if it doesn't exist)
conn = sqlite3.connect('reviews.sqlite')

# Create a cursor to navigate the database
c = conn.cursor()

# Select data entries between certain times
c.execute("SELECT * FROM review_db WHERE date BETWEEN '2017-01-01 10:10:10' AND DATETIME('now')")

# Read in the selected data and display
results = c.fetchall()
print(results)

# Close the connection to the database
conn.close()

