from os import listdir, remove
from os.path import isfile, join, splitext
import os
import requests
import face_recognition
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from dotenv import load_dotenv
import logging

# === Configuración inicial ===

# Cargar variables del .env
load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000")
RECOGNITION_THRESHOLD = float(os.getenv("MATCH_TOLERANCE", "0.6"))
LOG_FILE_PATH = os.getenv("LOG_FILE", "reconocimiento.log")

# Configurar logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Inicializar app Flask
app = Flask(__name__)
CORS(app)

# === Variables ===

faces_dict = {}  # (No usado si se consulta desde Laravel)
persistent_faces = "/root/faces"

# === Utilidades ===


def is_picture(filename):
    image_extensions = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    return [
        join(path, f) for f in listdir(path) if isfile(join(path, f)) and is_picture(f)
    ]


def remove_file_ext(filename):
    return splitext(filename.rsplit("/", 1)[-1])[0]


def calc_face_encoding(image):
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)
    if len(faces) > 1:
        raise Exception("Found more than one face in the image.")
    if not faces:
        raise Exception("No face found in the image.")
    return faces[0]


def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict(
        [(remove_file_ext(image), calc_face_encoding(image)) for image in image_files]
    )


def extract_image(request):
    if "file" not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files["file"]
    if file.filename == "":
        raise BadRequest("File is empty or invalid")
    return file


# === Llamadas a Laravel ===


def get_faces_from_laravel(matricula_id):
    url = f"{LARAVEL_API_URL}/api/biometricos/matricula/{matricula_id}"
    logging.info(f"Solicitando rostros para matrícula ID {matricula_id}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["rostros"]


def reportar_asistencias(matricula_id, rostros_detectados, timestamp):
    url = f"{LARAVEL_API_URL}/api/asistencias/registro-masivo"
    data = {
        "matricula_id": matricula_id,
        "rostros_detectados": rostros_detectados,
        "captura": timestamp,
    }
    logging.info(f"Enviando asistencias detectadas: {data}")
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        logging.info("✔ Asistencias registradas correctamente.")
        return True
    except Exception as e:
        logging.error(f"❌ Error al registrar asistencias: {str(e)}")
        return False


# === Lógica de reconocimiento ===


def detect_faces_in_image(file_stream, rostros_a_comparar):
    img = face_recognition.load_image_file(file_stream)
    uploaded_faces = face_recognition.face_encodings(img)

    logging.info(f"{len(uploaded_faces)} rostro(s) detectado(s) en imagen recibida.")

    rostros_detectados = []

    if uploaded_faces:
        for uploaded_face in uploaded_faces:
            for rostro in rostros_a_comparar:
                known_encoding = rostro["encoding"]
                match = face_recognition.compare_faces(
                    [known_encoding], uploaded_face, tolerance=RECOGNITION_THRESHOLD
                )[0]

                if match:
                    distancia = face_recognition.face_distance(
                        [known_encoding], uploaded_face
                    )[0]
                    rostros_detectados.append(
                        {"id": rostro["id"], "dist": float(distancia)}
                    )

    logging.info(f"{len(rostros_detectados)} coincidencias encontradas.")
    return {"count": len(uploaded_faces), "faces": rostros_detectados}


# === Endpoints ===


@app.route("/", methods=["POST"])
def web_recognize():
    file = extract_image(request)
    matricula_id = request.args.get("matricula_id")

    if not matricula_id:
        logging.error("Falta el parámetro 'matricula_id'.")
        raise BadRequest("Missing 'matricula_id' in query parameters")

    if file and is_picture(file.filename):
        logging.info(f"Inicio de proceso para matrícula {matricula_id}")
        rostros = get_faces_from_laravel(matricula_id)
        resultado = detect_faces_in_image(file, rostros)

        timestamp = datetime.now().isoformat()

        if resultado["faces"]:
            enviado = reportar_asistencias(matricula_id, resultado["faces"], timestamp)
            resultado["asistencia_reportada"] = enviado
        else:
            resultado["asistencia_reportada"] = False

        resultado["timestamp"] = timestamp
        logging.info(f"Resultado del proceso: {resultado}")
        return jsonify(resultado)

    raise BadRequest("Invalid file")


@app.route("/encoding", methods=["POST"])
def encode_face():
    file = extract_image(request)
    if file and is_picture(file.filename):
        try:
            encoding = calc_face_encoding(file)
            return jsonify({"encoding": encoding.tolist()})
        except Exception as e:
            logging.error(f"Error en encoding: {str(e)}")
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Invalid image"}), 400


@app.route("/faces", methods=["GET", "POST", "DELETE"])
def web_faces():
    if request.method == "GET":
        return jsonify(list(faces_dict.keys()))

    file = extract_image(request)
    if "id" not in request.args:
        raise BadRequest("Missing 'id' parameter!")

    if request.method == "POST":
        app.logger.info("%s loaded", file.filename)
        file.save(f"{persistent_faces}/{request.args.get('id')}.jpg")
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict.update({request.args.get("id"): new_encoding})
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == "DELETE":
        faces_dict.pop(request.args.get("id"))
        remove(f"{persistent_faces}/{request.args.get('id')}.jpg")

    return jsonify(list(faces_dict.keys()))


@app.route("/status", methods=["GET"])
def health_check():
    return (
        jsonify({"status": "ok", "message": "Face Recognition Service is running!"}),
        200,
    )


# === Main ===

if __name__ == "__main__":
    logging.info("Iniciando microservicio de reconocimiento facial")
    try:
        faces_dict = get_faces_dict(persistent_faces)
    except Exception as e:
        logging.warning(f"No se pudieron cargar rostros persistentes: {e}")
        faces_dict = {}

    logging.info("Servidor iniciado en puerto 8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
