# database_utils.py

import psycopg2
from config import PASSWORD, DB_HOST, DB_NAME, DB_PORT, DB_USER

DB_CONNECTION_PARAMS = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": PASSWORD,
    "host": DB_HOST,
    "port": DB_PORT  
}

def get_connection():
    return psycopg2.connect(**DB_CONNECTION_PARAMS)

def check_plate_in_database(plate_number):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT passno FROM registeredvehicles WHERE vehicleno = %s"
    cursor.execute(query, (plate_number,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        return True, result[0]
    return False, None

def add_visitor_entry(vehicle_no, visit_date, visit_time):
    conn = get_connection()
    cursor = conn.cursor()
    query = "INSERT INTO visitor (vehicleno, visitdate, visittime) VALUES (%s, %s, %s)"
    cursor.execute(query, (vehicle_no, visit_date, visit_time))
    conn.commit()
    cursor.close()
    conn.close()

def add_registered_vehicle(name, personal_no, pass_no, vehicle_no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM registeredvehicles WHERE vehicleno = %s", (vehicle_no,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return False  
    query = "INSERT INTO registeredvehicles (name, personalno, passno, vehicleno) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (name, personal_no, pass_no, vehicle_no))
    conn.commit()
    cursor.close()
    conn.close()
    return True 


def delete_registered_vehicle(vehicle_no):
    conn = get_connection()
    cursor = conn.cursor()
    query = "DELETE FROM registeredvehicles WHERE vehicleno = %s"
    cursor.execute(query, (vehicle_no,))
    conn.commit()
    cursor.close()
    conn.close()

def get_all_registered_vehicles():
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT name, personalno, passno, vehicleno FROM registeredvehicles"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_all_visitor_logs():
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT vehicleno, visitdate, visittime FROM visitor ORDER BY visitdate DESC, visittime DESC"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    visitor_logs = [
        (f"{row[1]} {row[2]}", row[0]) for row in rows
    ]
    return visitor_logs
