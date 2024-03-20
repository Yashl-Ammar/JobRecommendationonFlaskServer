install:
	pip3 install -r requirements.txt

docker:
	docker build -t flask-model-app .

images:
	docker images

run:
	docker run -p 5000:5000 flask-model-app

delete:
	docker rmi flask-model-app:latest 