import numpy as np
import cv2
from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=["http://localhost:9200"])

def extract_features(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    resized = cv2.resize(image, (128, 128))
    return resized.flatten() / 255.0

# def search_similar_images(image_file):
#     features = extract_features(image_file)
#     cluster = KMeans(n_clusters=10, random_state=0).fit([features])
#     cluster_id = int(cluster.predict([features])[0])

#     response = es.search(index="images", body={
#         "query": {
#             "term": {"cluster_id": cluster_id}
#         }
#     })
#     return [{"url": hit["_source"]["url"]} for hit in response["hits"]["hits"]]

def search_similar_images(image_file):
    features = extract_features(image_file)
    # ❌ Do not fit KMeans here with one image
    # cluster = KMeans(n_clusters=10).fit([features])
    # cluster_id = int(cluster.predict([features])[0])

    cluster_id = 3  # ✅ For now, just assume we're looking for cluster 3
    response = es.search(index="images", body={
        "query": {
            "term": {"cluster_id": cluster_id}
        }
    })
    return [{"url": hit["_source"]["url"]} for hit in response["hits"]["hits"]]
