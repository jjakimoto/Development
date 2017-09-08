import scrapy
from scrapy import Item, Field

class RedditScrapyItem(Item):
    subreddit = Field()
    link = Field()
    title  = Field()
    date = Field()
    html = Field()