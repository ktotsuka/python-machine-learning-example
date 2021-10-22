import sqlite3
import os
import numpy as np

# Create a connection to SQLite database (it creates a database if it doesn't exist)
conn = sqlite3.connect('reviews.sqlite')

# Create a cursor to navigate the database
c = conn.cursor()

# Delete "review_db" table if it exists
c.execute('DROP TABLE IF EXISTS review_db')

# Create "review_db" table
c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')

# Insert two example reviews into the table
example1 = 'I love this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (example2, 0))

# Save the changes
conn.commit()

# Close connection to the database
conn.close()