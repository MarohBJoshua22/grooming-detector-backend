from flask import Flask, request, jsonify 

app = Flask(__name__)

@app.route("/")
def home():
  return jsonify({'name': 'Maroh','email': 'maroh@outlook.com', 'phone': '07543167969'})

 

if __name__ == "__main__":
  app.run(debug=True)

 