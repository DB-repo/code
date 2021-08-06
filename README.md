# Data management systems for query time data enrichment 

The following instructions are for installing the EQ_TC implementation


```
git clone <git repo link>
sudo docker build -t tdb db-master/
sudo docker run -p 5000:5000 --name db_con tdb
```

Connect to  the PostgreSQL server by running tdb inside docker 
```
sudo docker exec -it db_con bash
su - postgres 
/usr/local/pgsql/bin/psql test
```
Run UI to execute and visualize queries

```
cd tdb/ui
python server.py  # requires python2

UI is accessible at localhost:8000
```
** This code is based on Apache MadLib.


The following instruction is for executing EQ_LC implementation

1) Install a PostgreSQL database server. 
2) Load the data to the PostgreSQL database using the script files in EQ_LC/sql directory.
3) Execute the script of ExecuteLC.py to execute the queries on the Tweet dataset. 



