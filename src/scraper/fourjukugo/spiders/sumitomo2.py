import scrapy
from fourjukugo.items import Post
from itertools import chain


class Sumitomo2Spider(scrapy.Spider):
    name = "sumitomo2"
    allowed_domains = ["cam.sumitomolife.co.jp"]
    start_urls = ["https://cam.sumitomolife.co.jp/jukugo/2020/nyusen.html"]

    def parse(self, response):
        titles = response.xpath(
            '//*[@id="main_article"]/div/div[2]/div/div/section/div/div/div/p/img/@alt'
        ).extract()
        meanings = response.xpath(
            '//*[@id="main_article"]/div/div[2]/div/div/section/div/div/div/div/div/p/text()'
        ).extract()
        print(titles)
        for title, meaning in zip(titles, meanings):
            print(title, meaning)
            yield Post(title=title, meaning=meaning, source="")
