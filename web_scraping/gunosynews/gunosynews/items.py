# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class GunosynewsItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    subcategory = scrapy.Field()
