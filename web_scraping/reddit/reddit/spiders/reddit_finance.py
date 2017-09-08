# -*- coding: utf-8 -*-
import scrapy


class RedditFinanceSpider(scrapy.Spider):
    name = 'reddit_finance'
    allowed_domains = ['https://www.reddit.com/r/finance']
    start_urls = ["https://www.reddit.com/r/finance/?count=21&after=t3_6snq8b"]

    def parse(self, response):
        for sel in response.css("div.top-matter"):
            article = RedditItem()
            article['title'] = sel.css("p.title > a::text").extract_first()
            article['url'] = sel.css("p.title > a::attr('href')").extract_first()
            # article['subcategory'] = sel.css("div.list_text > a::text").extract_first()
            yield article

        next_page = response.css("div.page-link-option > a::attr('href')")
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield scrapy.Request(url, callback=self.parse)
