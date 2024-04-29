BASE_URL=http://127.0.0.1:5000
SPEAKER_NAME=
TRANSCRIPT=
FILE_PATH=

upload:
	curl -X 'POST' \
	'$(BASE_URL)/add_speaker' \
	-H 'accept: application/json' \
	-H 'Content-Type: multipart/form-data' \
	-F 'name=$(SPEAKER_NAME)' \
	-F 'transcript="$(TRANSCRIPT)"' \
	-F 'file=@$(FILE_PATH)'


lint:
    black api
