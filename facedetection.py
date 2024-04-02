import streamlit as st
import cv2
import numpy as np
import sqlite3
import face_recognition

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def display_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('face_database.db')
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      image_name TEXT UNIQUE,
                      name TEXT,
                      age TEXT,
                      number TEXT,
                      email TEXT,
                      face_embedding BLOB)''')
    except sqlite3.Error as e:
        print(e)

def insert_face(conn, image_name, name, age, number, email, face_encoding):
    try:
        c = conn.cursor()
        c.execute('''INSERT INTO faces (image_name, name, age, number, email, face_embedding)
                     VALUES (?, ?, ?, ?, ?, ?)''', (image_name, name, age, number, email, sqlite3.Binary(face_encoding)))
        conn.commit()
    except sqlite3.IntegrityError:
        print("Image already exists in the database.")


def get_face_details(conn, image_name):
    try:
        c = conn.cursor()
        c.execute('''SELECT name, age, number, email FROM faces WHERE image_name = ?''', (image_name,))
        row = c.fetchone()
        if row:
            return {'name': row[0], 'age': row[1], 'number': row[2], 'email': row[3]}
        else:
            return None
    except sqlite3.Error as e:
        print(e)
        return None

def get_face_details_by_embedding(conn, face_embedding):
    try:
        c = conn.cursor()
        c.execute('''SELECT * FROM faces''')
        rows = c.fetchall()
        for row in rows:
            saved_face_embedding = np.frombuffer(row[6], dtype=np.float64) # Adjust the index as per your table structure
            if np.array_equal(face_embedding, saved_face_embedding):
                return {'name': row[2], 'age': row[3], 'number': row[4], 'email': row[5]}
        return None
    except sqlite3.Error as e:
        print(e)
        return None



def main():
    conn = create_connection()
    if conn is not None:
        create_table(conn)
    else:
        st.error("Failed to connect to the database.")

    st.title("Face Detection and Recognition")

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, -1)

        faces = detect_faces(image)

        if len(faces) == 0:
            st.error("Face not detected. Please upload an image with a clear face.")
        else:
            st.success("Face detected!")

            face_encodings = face_recognition.face_encodings(image, faces)  # Using face_recognition

            # Check for matches in the database using face embeddings
            matched_face_details = None
            for face_encoding in face_encodings:
                matched_face_details = get_face_details_by_embedding(conn, face_encoding)
                if matched_face_details:
                    break  # Found a match, stop searching

            if matched_face_details:
                st.subheader("Face Found in Database")
                st.write("Name:", matched_face_details['name'])
                st.write("Age:", matched_face_details['age'])
                st.write("Number:", matched_face_details['number'])
                st.write("Email:", matched_face_details['email'])
            else:
                st.subheader("Face not found in the database. Please provide details:")
                name = st.text_input("Name")
                age = st.text_input("Age")
                number = st.text_input("Number")
                email = st.text_input("Email")
                if st.button("Save"):
                    for face_encoding in face_encodings:
                        insert_face(conn, uploaded_file.name, name, age, number, email, face_encoding)

    conn.close()

if __name__ == "__main__":
    main()
