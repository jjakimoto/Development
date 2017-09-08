# -*- coding: utf-8 -*-
import scrapy
import re
from scrapy import Request

from bs4 import BeautifulSoup

from reddit_scrapy.items import RedditScrapyItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from urllib.request import urlopen


class RedditSpider(scrapy.Spider):
    name = 'reddit'
    allowed_domains = ['reddit.com']
    start_urls =[# 'https://www.reddit.com/r/circlejerk', 
                # 'https://www.reddit.com/r/gaming', 
                # 'https://www.reddit.com/r/floridaman',  
                # 'https://www.reddit.com/r/movies', 
                # 'https://www.reddit.com/r/science', 
                # 'https://www.reddit.com/r/seahawks', 
                # 'https://www.reddit.com/r/totallynotrobots', 
                # 'https://www.reddit.com/r/uwotm8', 
                # 'https://www.reddit.com/r/videos', 
                'https://www.reddit.com/r/worldnews']

    def parse(self, response):
        links = response.xpath('//p[@class="title"]/a[@class="title may-blank outbound"]/@href').extract()
        dates = response.xpath('//p[@class="tagline "]/time[@class="live-timestamp"]/@title').extract()
        titles = response.xpath('//p[@class="title"]/a[@class="title may-blank outbound"]/text()').extract()
        
    
        # Store data
        for title, link in zip(titles, links):
            item = RedditScrapyItem()
            item['subreddit'] = str(re.findall('/r/[A-Za-z]*8?', link))[3:len(str(re.findall('/r/[A-Za-z]*8?', link))) - 2]
            link = response.urljoin(link)
            item['link'] = link
            item['title'] = title
            print("**************************", link)
            try:
                html =  urlopen(link).read()
                html = BeautifulSoup(html)
                item["html"] = html.get_text()
            except:
                pass
            # item["html"] = urlopen(link).read()
            yield item