from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

@app.before_first_request
def setupModel():
    print("Setting up model")
    # Load model here

class SemanticSeg(Resource):
    def get(self):
        return {'hello': 'world'}

class getNearestNeighbours(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(SemanticSeg, '/segmentation')
api.add_resource(getNearestNeighbours, '/nearestNeighbours')

if __name__ == '__main__':
    app.run()