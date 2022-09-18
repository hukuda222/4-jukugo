import scrapy


class QiitaSpider(scrapy.Spider):
    name = 'qiita'
    allowed_domains = ['qiita.com']
    start_urls = ['https://qiita.com/']

    def parse(self, response):
        datas = response.xpath('/html/body/div[1]/div[1]/div[2]/div/a/@href').extract()
        print(datas)
