
import requests

url = "https://storage.googleapis.com/kagglesdsdata/datasets/118250/283795/cifar10/test/airplane/0001.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250118T183347Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=77448fe57620956fac2aba49973f33eef7b9d4406d57e3d91b712a4c586f59a029b18a8d650c36667a8c4da00d2bcf2e62ec6a243121edc3f86399bf2c0cb10df3347d0c44c48411491beebb14c65edc79f5425f16043644aa5d5304a0dddf036322ea211dfbd511c72a735b99df181a77eedba21826c2508c27b07a90d0a793a8482f264c73f9e0eb5f75055e08e65e8b86c93259221829e43788f5c417d7d11b50c643c414ba44ae9cf839850882b3cb1db8bee488fd52041e2ee7ed63c0150ab383c4c62be764859fd3121c8f9541a8a34ed1f4aa10a6979f49ed691c30eb30337d5b8d4f2433a66625dc842407012ddc76a771bfcb971879ca308f898363"
host_url = "http://34.32.217.62:9696/predict"
data = {'url':url}
response = requests.post(host_url,json=data).json()
print(response)
