from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json

import time
import random


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        # Done
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.crawled_ids = []
        self.added_ids = []
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        # Done
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        with open('IMDB_crawled2.json', 'w') as f:
            f.write(json.dumps(self.crawled))
            f.close()

        with open('IMDB_not_crawled2.json', 'w') as f:
            f.write(json.dumps(self.not_crawled))
            f.close()

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        # Done
        with open('IMDB_crawled2.json', 'r') as f:
            raw_data_0 = f.read()
            f.close()

        self.crawled = json.loads(raw_data_0)

        with open('IMDB_not_crawled2.json', 'w') as f:
            raw_data_1 = f.read()
            f.close()

        self.not_crawled = json.loads(raw_data_1)

        self.added_ids = [self.get_id_from_URL(link) for link in self.not_crawled]

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
        ]
        IMDbCrawler.headers['User-Agent'] = random.choice(user_agents)
        res = get(URL, headers=IMDbCrawler.headers)
        if res.status_code == 200:
            return res
        else:
            return None


    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        # Done
        res = self.crawl(self.top_250_URL)
        top_soup = BeautifulSoup(res.content, 'html.parser')
        link_elements = top_soup.select('a[href]')
        abs = 'https://www.imdb.com/title/'
        for link in link_elements:
            if link['href'].startswith('/title'):
                id = link['href'].split('/')[2]
                new_url = abs + id
                if new_url not in self.not_crawled:
                    self.not_crawled.append(new_url)
                    self.added_ids.append(id)


    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO:
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        WHILE_LOOP_CONSTRAINTS = None
        NEW_URL = None
        THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while len(self.crawled) < self.crawling_threshold:

                self.add_list_lock.acquire()
                URL = self.not_crawled.pop(0)
                self.add_list_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                if len(self.not_crawled) == 0:
                    wait(futures)
                    # print(futures[0].result())
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print(f"start crawling {URL}")
        # TODO

        if self.get_id_from_URL(URL) in self.crawled_ids:
          return

        movie_id = self.get_id_from_URL(URL)
        review_URL = self.get_review_link(URL)
        summary_URL = self.get_summary_link(URL)
        mpaa_URL = URL + '/parentalguide'

        res = self.crawl(URL)
        time.sleep(2)
        summary_res = self.crawl(summary_URL)
        time.sleep(2)
        review_res = self.crawl(review_URL)
        time.sleep(2)
        mpaa_res = self.crawl(mpaa_URL)
        time.sleep(2)

        # print(res is not None, summary_res is not None, review_res is not None, mpaa_res is not None)
        if (res is not None) \
        and (summary_res is not None) \
        and (review_res is not None) \
            and (mpaa_res is not None):
            movie_instance = self.get_imdb_instance()
            self.extract_movie_info([res, summary_res, review_res, mpaa_res],
                                     movie_instance, URL)

            self.add_list_lock.acquire()
            self.crawled.append(movie_instance)
            self.added_ids.remove(movie_id)
            self.crawled_ids.append(movie_id)

            for link in movie_instance['related_links']:
                self.not_crawled.append(link)
                self.added_ids.append(self.get_id_from_URL(link))

            self.add_list_lock.release()

            print('successful', movie_instance['title'])
            return movie_instance
        else:
            self.add_list_lock.acquire()
            self.not_crawled.append(URL)
            self.add_list_lock.release()
            print('failed')

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO

        main_res, summary_res, review_res, mpaa_res = res
        main_soup = BeautifulSoup(main_res.content, 'html.parser')
        summary_soup = BeautifulSoup(summary_res.content, 'html.parser')
        review_soup = BeautifulSoup(review_res.content, 'html.parser')
        mpaa_soup = BeautifulSoup(mpaa_res.content, 'html.parser')

        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(main_soup)
        movie['first_page_summary'] = self.get_first_page_summary(main_soup)
        movie['release_year'] = self.get_release_year(main_soup)
        movie['mpaa'] = self.get_mpaa(mpaa_soup)
        movie['budget'] = self.get_budget(main_soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(main_soup)
        movie['directors'] = self.get_director(main_soup)
        movie['writers'] = self.get_writers(main_soup)
        movie['stars'] = self.get_stars(main_soup)
        movie['related_links'] = self.get_related_links(main_soup)
        movie['genres'] = self.get_genres(main_soup)
        movie['languages'] = self.get_languages(main_soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(main_soup)
        movie['rating'] = self.get_rating(main_soup)
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            # TODO
            # Done
            return url + '/plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            # TODO
            # Done
            return url + '/reviews'
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            # TODO
            # Done
            name = soup.find('span', class_ = 'hero__primary-text').text
            return name
        except:
            print("failed to get title")
            return ''

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            # TODO
            # Done
            summary = soup.find('span', class_ = 'sc-466bb6c-0').text
            return summary
        except:
            print("failed to get first page summary")
            return ''

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # TODO
            # Done
            directors = []
            sub_sections = soup.find_all(class_ = 'ipc-metadata-list__item')
            for section in sub_sections:
                parts = section.find_all(class_ = 'ipc-metadata-list-item__label')
                exist = False
                for part in parts:
                    try:
                        tex = part.text
                    except:
                        continue

                    if tex == 'Director' or tex == 'Directors':
                        exist = True
                        break
                if exist:
                    names = section.find_all(class_ = 'ipc-metadata-list-item__list-content-item')
                    for name in names:
                         if name.text not in directors:
                          directors.append(name.text)


            return directors
        except:
            print("failed to get director")
            return []

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            # TODO
            # Done
            sub_sections = soup.find_all(class_ = 'ipc-metadata-list__item')
            stars = []
            for section in sub_sections:
                parts = section.find_all(class_ = 'ipc-metadata-list-item__label')
                exist = False
                for part in parts:
                    try:
                        tex = part.text
                    except:
                        continue

                    if tex == 'Stars' or tex == 'Star':
                        exist = True
                        break

                if exist:
                    names = section.find_all(class_ = 'ipc-metadata-list-item__list-content-item')
                    for name in names:
                        if name.text not in stars:
                          stars.append(name.text)

            return stars
        except:
            print("failed to get stars")
            return []

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # TODO
            # Done
            sub_sections = soup.find_all(class_ = 'ipc-metadata-list__item')

            writers = []
            for section in sub_sections:
                parts = section.find_all(class_ = 'ipc-metadata-list-item__label')
                exist = False
                for part in parts:
                    try:
                        tex = part.text
                    except:
                        continue

                    if tex == 'Writer' or tex == 'Writers':
                         exist = True
                         break
                if exist:
                    names = section.find_all(class_ = 'ipc-metadata-list-item__list-content-item')
                    for name in names:
                        if name.text not in writers:
                          writers.append(name.text)

            return writers
        except:
            print("failed to get writers")
            return []

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('section', class_ = 'ipc-page-section ipc-page-section--base celwidget')
            abs = 'https://www.imdb.com/title/'
            related_links = []
            for x in A:
                a = x.find_all('span')
                for b in a:
                    if b.text == 'More like this':
                        c = x.find_all('a', class_ = 'ipc-lockup-overlay ipc-focusable')
                        for d in c:
                            id = d['href'].split('/')[2]
                            link = abs + id
                            related_links.append(link)

            return related_links
        except:
            print("failed to get related links")
            return []

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            # Done
            sections = soup.find_all('section', class_ = 'ipc-page-section')
            summary_list = []
            for section in sections:
                spans = section.find_all('span')
                exist = False
                for span in spans:
                    try:
                        tex = span.text
                    except:
                        continue

                    if tex == 'Summaries':
                        exist = True
                        break

                if exist:
                    summaries = section.find_all('div', class_ = 'ipc-html-content-inner-div')
                    for sum in summaries:
                        summary_list.append(sum.text)

            return summary_list
        except:
            print("failed to get summary")
            return []

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            # TODO
            # Done
            sections = soup.find_all('section', class_ = 'ipc-page-section')
            synopsis_list = []
            for section in sections:
                spans = section.find_all('span')
                exist = False
                for span in spans:
                    try:
                        tex = span.text
                    except:
                        continue
                    if tex == 'Synopsis':
                        exist = True
                        break

                if exist:
                    summaries = section.find_all('div', class_ = 'ipc-html-content-inner-div')
                    for sum in summaries:
                        synopsis_list.append(sum.text)

            return synopsis_list
        except:
            print("failed to get synopsis")
            return []

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            # Done
            reviews = soup.find_all('div', class_ = 'review-container')
            reviews_list = []
            for review in reviews:
                score_part = review.find('span', class_ = 'rating-other-user-rating')
                if score_part is not None:
                    score = score_part.find('span').text
                else:
                    score = ''
                review_text = review.find('div', class_ = 'show-more__control').text
                reviews_list.append([review_text, score])

            return reviews_list
        except:
            print("failed to get reviews")
            return []

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            # Done
            genres = soup.find_all('span', class_ = 'ipc-chip__text')
            genre_list = []
            for genre in genres:
                g = genre.text
                if not g.startswith('Back'):
                    genre_list.append(g)

            return genre_list
        except:
            print("Failed to get generes")
            return []

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('div', class_ = 'sc-acdbf0f3-0 haeNPA rating-bar__base-button')
            rating = ''
            for x in A:
                a = x.find_all('div', class_ = 'sc-acdbf0f3-1 kCTJoV')
                for b in a:
                    if b.text == 'IMDb RATING':
                        c = x.find_all('span', class_ = 'sc-bde20123-1 cMEQkK')
                        for d in c:
                            rating = d.text
                            break

            return rating
        except:
            print("failed to get rating")
            return []

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            # Done
            mpaa = soup.find_all('td')
            MPAA = ''
            if mpaa[0].text == 'MPAA':
                MPAA = mpaa[1].text

            return MPAA
        except:
            print("failed to get mpaa")
            return ''

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            # Done
            part = soup.find(class_ = 'sc-67fa2588-0')
            year = part.find(class_ = 'ipc-link').text
            return year
        except:
            print("failed to get release year")
            return ''

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('li', class_ = 'ipc-metadata-list__item')
            languages = []
            for x in A:
                a = x.find_all('span', class_ = 'ipc-metadata-list-item__label')
                for b in a:
                    try:
                        st = b.text
                    except:
                        continue
                    if st == 'Languages' or st == 'Language':
                        c = x.find_all('a', class_ = 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
                        for d in c:
                            languages.append(d.text)

            return languages
        except:
            print("failed to get languages")
            return []

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('li', class_ = 'ipc-metadata-list__item')
            countries = []
            for x in A:
                a = x.find_all('span', class_ = 'ipc-metadata-list-item__label')
                for b in a:
                    try:
                        st = b.text
                    except:
                        continue
                    if st == 'Country of origin' or st == 'Countries of origin':
                        c = x.find_all('a', class_ = 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
                        for d in c:
                            countries.append(d.text)
            return countries
        except:
            print("failed to get countries of origin")
            return []

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('li', class_ = 'ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT')
            budget = ''
            for x in A:
                a = x.find_all('span', class_ = 'ipc-metadata-list-item__label')
                for b in a:
                    try:
                        st = b.text
                    except:
                        continue
                    if st == 'Budget':
                        c = x.find_all('span', class_ = 'ipc-metadata-list-item__list-content-item')
                        for d in c:
                            budget = d.text

            return budget
        except:
            print("failed to get budget")
            return ''

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            # TODO
            # Done
            A = soup.find_all('li', class_ = 'ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT')
            gross_worldwide = ''
            for x in A:
                a = x.find_all('span', class_ = 'ipc-metadata-list-item__label')
                for b in a:
                    try:
                        st = b.text
                    except:
                        continue
                    if st == 'Gross worldwide':
                        c = x.find_all('span', class_ = 'ipc-metadata-list-item__list-content-item')
                        for d in c:
                            gross_worldwide = d.text

            return gross_worldwide
        except:
            print("failed to get gross worldwide")
            return ''


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()

if __name__ == '__main__':
  main()
