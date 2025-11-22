IMAGE_NAME = autoenconder:latest

build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run --rm $(IMAGE_NAME)

re: build run

shell:
	docker run -it --rm $(IMAGE_NAME) sh

clean:
	docker rmi $(IMAGE_NAME)
