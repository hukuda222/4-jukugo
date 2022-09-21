import scrapy
from fourjukugo.items import Post
from itertools import chain


class SanaboSpider(scrapy.Spider):
    name = "sanabo"
    allowed_domains = ["sanabo.com"]
    start_urls = ["http://sanabo.com/words"]

    def parse(self, response):
        urls = response.xpath(
            '//*[@id="custom_html-15"]/div/center/table/tr/td/a/@href'
        ).extract()
        return [scrapy.Request(url, callback=self.parse_1char) for url in urls]

    def parse_1char(self, response):
        urls = response.xpath('//*[@id="main"]/div[2]/article/a/@href').extract()
        add_urls = response.xpath('///*[@id="main"]/nav/ul/li/a/@href').extract()
        return [scrapy.Request(url, callback=self.parse_4char) for url in urls] + [
            scrapy.Request(url, callback=self.parse_1char) for url in add_urls
        ]

    def parse_4char(self, response):
        title = response.xpath("/html/body/div[1]/div[2]/ul/li/text()").extract()
        meaning = response.xpath(
            "/html/body/div[1]/div[3]/div/main/article/section/ul//text()"
        ).extract()[1:2]
        source = response.xpath(
            "/html/body/div[1]/div[3]/div/main/article/section/div[2]/ul/li[1]/text()"
        ).extract()
        print("3", title, meaning, source)
        return Post(title=title, meaning=meaning, source=source)
