import scrapy
from fourjukugo.items import Post
from itertools import chain


class SumitomoSpider(scrapy.Spider):
    name = "sumitomo"
    allowed_domains = ["cam.sumitomolife.co.jp"]
    start_urls = ["https://cam.sumitomolife.co.jp/jukugo/30th.html"]

    def parse(self, response):
        titles = []
        meanings = []
        for y in range(1990, 2020):
            for i in range(1, 3):
                for j in range(1, 6):
                    title = response.xpath(
                        '//*[@id="anc-'
                        + str(y)
                        + '"]/div[2]/div['
                        + str(i)
                        + "]/table/tbody/tr["
                        + str(j)
                        + "]/td[1]/strong/text()"
                    ).extract()
                    meaning = response.xpath(
                        '//*[@id="anc-'
                        + str(y)
                        + '"]/div[2]/div['
                        + str(i)
                        + "]/table/tbody/tr["
                        + str(j)
                        + "]/td[2]/text()"
                    ).extract()
                    titles.append(title)
                    meanings.append(meaning)
        for title, meaning in zip(titles, meanings):
            yield Post(title=title, meaning=meaning, source="")
