import sqlite3

conn = sqlite3.connect('test.db')
print('Opened database successfully')

conn.execute(
    '''
    CREATE TABLE STUDENT(
        REGNO TEXT PRIMARY KEY NOT NULL,
        NAME TEXT NOT NULL,
        AGE INT NOT NULL,
        SEM CHAR(50),
        BRANCH TEXT
        );
    '''
)

print('Table created successfully')
conn.execute(
    '''
    INSERT INTO STUDENT (REGNO,NAME,AGE,SEM,BRANCH) 
    VALUES ('1', 'Paul', 32, 'VI', 'IT' )
    '''
)
conn.execute(
    '''
    INSERT INTO STUDENT (REGNO,NAME,AGE,SEM,BRANCH)
    VALUES ('2', 'Allen', 25, 'V', 'CSE' )
    '''
)
conn.execute(
    '''
    INSERT INTO STUDENT (REGNO,NAME,AGE,SEM,BRANCH)
    VALUES ('3', 'Teddy', 23, 'III', 'CCE' )
    '''
)
conn.execute(
    '''
    INSERT INTO STUDENT (REGNO,NAME,AGE,SEM,BRANCH)
    VALUES ('4', 'Mark', 25, 'VIII', 'ECE' )
    '''
)
conn.commit()
print('Records created successfully')

cursor = conn.execute('SELECT * FROM STUDENT WHERE REGNO="2" OR REGNO="3"')

for row in cursor:
    print('REGNO = ', row[0])
    print('NAME = ', row[1])
    print('AGE = ', row[2])
    print('SEM = ', row[3])
    print('BRANCH = ', row[4], end = '\n\n')

print('Operation done successfully')
conn.close()

