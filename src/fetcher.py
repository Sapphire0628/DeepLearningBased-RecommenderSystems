import utils


def main():
    path = './image/'
    gzip_path = './data/meta_AMAZON_FASHION.json.gz'
    url_col = ["imageURLHighRes", "imageURL"]
    uid_col = "asin"
    utils.get_image(path, gzip_path, url_col, uid_col)


if __name__ == "__main__":
    main()
