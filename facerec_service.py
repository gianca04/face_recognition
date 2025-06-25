from os import listdir, remove
from os.path import isfile, join, splitext

import requests

import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Diccionario global que almacena los rostros conocidos (identificador -> encoding facial)
faces_dict = {}

# Ruta en el sistema donde se almacenan de forma persistente las imágenes de rostros
persistent_faces = "/root/faces"

# Crear la aplicación Flask y habilitar CORS para permitir peticiones desde otros orígenes
app = Flask(__name__)
CORS(app)

# Obtener todos los rostros de una matricula
def get_faces_from_laravel(matricula_id):
    url = f"http://attendance_api/api/biometricos/matricula/{matricula_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["rostros"]

# <Funciones para manejo de imágenes> #

# Verifica si el archivo tiene una extensión válida de imagen
def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

# Devuelve una lista con rutas completas de archivos de imagen en el directorio especificado
def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]

# Elimina la extensión del archivo y devuelve solo el nombre base
def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

# Obtiene el encoding facial (vector de características) de la primera cara en una imagen
def calc_face_encoding(image):
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    if len(faces) > 1:
        raise Exception("Found more than one face in the given training image.")

    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]  # Devuelve el encoding de la única cara encontrada

# Construye un diccionario con ID de persona y su encoding facial a partir de imágenes almacenadas
def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict([
        (remove_file_ext(image), calc_face_encoding(image))
        for image in image_files
    ])


# Detecta rostros en una imagen subida y compara con los rostros conocidos
def detect_faces_in_image(file_stream, rostros_a_comparar):
    img = face_recognition.load_image_file(file_stream)
    uploaded_faces = face_recognition.face_encodings(img)

    faces_found = len(uploaded_faces)
    faces = []

    if faces_found:
        for uploaded_face in uploaded_faces:
            for rostro in rostros_a_comparar:
                known_encoding = rostro["encoding"]
                match_result = face_recognition.compare_faces([known_encoding], uploaded_face, tolerance=0.6)[0]

                if match_result:
                    distancia = face_recognition.face_distance([known_encoding], uploaded_face)[0]
                    faces.append({
                        "id": rostro["id"],
                        "dist": float(distancia)
                    })

    return {
        "count": faces_found,
        "faces": faces
    }


# <Funciones para manejo de imágenes> #

# <Controladores HTTP> #

# Ruta principal para reconocimiento facial desde una imagen enviada por POST
@app.route('/', methods=['POST'])
def web_recognize():
    file = extract_image(request)
    matricula_id = request.args.get("matricula_id")

    if not matricula_id:
        raise BadRequest("Missing 'matricula_id' in query params")

    if file and is_picture(file.filename):
        rostros = get_faces_from_laravel(matricula_id)
        return jsonify(detect_faces_in_image(file, rostros))
    else:
        raise BadRequest("Given file is invalid!")

# Ruta para manejar imágenes persistentes (registro y eliminación de rostros)
@app.route('/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    # Retorna una lista de IDs de rostros registrados
    if request.method == 'GET':
        return jsonify(list(faces_dict.keys()))

    # Extrae la imagen enviada
    file = extract_image(request)
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    # Registrar un nuevo rostro
    if request.method == 'POST':
        app.logger.info('%s loaded', file.filename)
        file.save("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict.update({request.args.get('id'): new_encoding})
        except Exception as exception:
            raise BadRequest(exception)

    # Eliminar un rostro por su ID
    elif request.method == 'DELETE':
        faces_dict.pop(request.args.get('id'))
        remove("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))

    return jsonify(list(faces_dict.keys()))

@app.route('/encoding', methods=['POST'])
def encode_face():
    file = extract_image(request)
    if file and is_picture(file.filename):
        try:
            encoding = calc_face_encoding(file)
            return jsonify({"encoding": encoding.tolist()})  # encoding es un ndarray → convertir a lista JSON
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Invalid image"}), 400


# Extrae la imagen del request y valida su existencia
def extract_image(request):
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file

# </Controladores HTTP> #

# Código que se ejecuta al iniciar el servidor
if __name__ == "__main__":
    print("Starting by generating encodings for found images...")
    # Carga todos los rostros persistentes al iniciar el servicio
    faces_dict = get_faces_dict(persistent_faces)
    print(faces_dict)

    # Inicia el servidor Flask en el puerto 8080, accesible desde cualquier IP
    print("Starting WebServer...")
    app.run(host='0.0.0.0', port=8080, debug=False)
