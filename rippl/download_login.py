import requests

class DownloadLogin(requests.Session):

    def __init__(self, host, username, password):

        super().__init__()
        self.auth = (username, password)
        self.host = host
        self.same_auth_count = 0

    # Overrides from the library to keep headers when redirected to or from the NASA auth host.
    def rebuild_auth(self, prepared_request, response):

        if self.same_auth_count == 0:
            print('Using the same authentication for redirected url')
        self.same_auth_count += 1

        return

    def download_file(self, url, filename, size=0):
        # Actual download of the file

        response = self.get(url, stream=True)
        print(response.status_code)

        # raise an exception in case of http errors
        response.raise_for_status()

        # save the file
        if filename:
            with open(filename, 'wb') as fd:
                mb_count = 0

                for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                    fd.write(chunk)
                    mb_count += 10
                    print(str(mb_count) + ' of ' + str(size) + ' MB downloaded from ' + filename)

        return True
