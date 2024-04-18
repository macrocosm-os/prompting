import bittensor as bt
import aiohttp
import asyncio
from datetime import datetime
from wikipedia import USER_AGENT, RATE_LIMIT, RATE_LIMIT_MIN_WAIT, API_URL, ODD_ERROR_MESSAGE, WikipediaPage, DisambiguationError, PageError
from wikipedia.exceptions import PageError, RedirectError
from bs4 import BeautifulSoup

async def _async_wiki_request(params):
    global RATE_LIMIT_LAST_CALL
    global USER_AGENT

    params['format'] = 'json'
    if 'action' not in params:
        params['action'] = 'query'

    headers = {'User-Agent': USER_AGENT}

    if RATE_LIMIT and RATE_LIMIT_LAST_CALL and \
        RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT > datetime.now():

        wait_time = (RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT) - datetime.now()
        await asyncio.sleep(wait_time.total_seconds())

    # bt.logging.info('*' * 15)
    # bt.logging.info('Querying Wikipedia API with params: {}'.format(params))
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params=params, headers=headers) as response:
            if RATE_LIMIT:
                RATE_LIMIT_LAST_CALL = datetime.now()
            return await response.json()
        
        
class AsyncWikipediaPage(WikipediaPage):
    def __init__(self, title=None, pageid=None, redirect=True, preload=False, original_title=''):
        if title is not None:
            self.title = title
            self.original_title = original_title or title
        elif pageid is not None:
            self.pageid = pageid
        else:
            raise ValueError("Either a title or a pageid must be specified")        
        
        self.redirect = redirect
        self.preload = preload

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.load())

        if preload:
            for prop in ('content', 'summary', 'images', 'references', 'links', 'sections'):
                getattr(self, prop)                            
    
    async def load(self):
        '''
        Load basic information from Wikipedia.
        Confirm that page exists and is not a disambiguation/redirect.

        Does not need to be called manually, should be called automatically during __init__.
        '''
        query_params = {
        'prop': 'info|pageprops',
        'inprop': 'url',
        'ppprop': 'disambiguation',
        'redirects': '',
        }
        if not getattr(self, 'pageid', None):
            query_params['titles'] = self.title
        else:
            query_params['pageids'] = self.pageid

        request = await _async_wiki_request(query_params)

        query = request['query']
        pageid = list(query['pages'].keys())[0]
        page = query['pages'][pageid]

        # missing is present if the page is missing
        if 'missing' in page:
            if hasattr(self, 'title'):
                raise PageError(self.title)
            else:
                raise PageError(pageid=self.pageid)

        # same thing for redirect, except it shows up in query instead of page for
        # whatever silly reason
        elif 'redirects' in query:
            if self.redirect:
                redirects = query['redirects'][0]

                if 'normalized' in query:
                    normalized = query['normalized'][0]
                    assert normalized['from'] == self.title, ODD_ERROR_MESSAGE

                    from_title = normalized['to']

                else:
                    from_title = self.title

                assert redirects['from'] == from_title, ODD_ERROR_MESSAGE

                # change the title and reload the whole object
                self.title = redirects['to']
                await self.load(redirects['to'], redirect=self.redirect, preload=self.preload)

            else:
                raise RedirectError(getattr(self, 'title', page['title']))

        # since we only asked for disambiguation in ppprop,
        # if a pageprop is returned,
        # then the page must be a disambiguation page
        elif 'pageprops' in page:
            query_params = {
                'prop': 'revisions',
                'rvprop': 'content',
                'rvparse': '',
                'rvlimit': 1
            }
            if hasattr(self, 'pageid'):
                query_params['pageids'] = self.pageid
            else:
                query_params['titles'] = self.title
            request = await _async_wiki_request(query_params)
            html = request['query']['pages'][pageid]['revisions'][0]['*']

            lis = BeautifulSoup(html, 'html.parser').find_all('li')
            filtered_lis = [li for li in lis if not 'tocsection' in ''.join(li.get('class', []))]
            may_refer_to = [li.a.get_text() for li in filtered_lis if li.a]

            raise DisambiguationError(getattr(self, 'title', page['title']), may_refer_to)

        else:
            self.pageid = pageid
            self.title = page['title']
            self.url = page['fullurl']
        
    
    @property
    async def sections(self):
        '''
        Overwrites the `sections` property to be async.
        '''
        if not getattr(self, '_sections', False):
            query_params = {
            'action': 'parse',
            'prop': 'sections',
            }
            if not getattr(self, 'title', None) is None:
                query_params["page"] = self.title

            request = await _async_wiki_request(query_params)
            self._sections = [section['line'] for section in request['parse']['sections']]

        return self._sections
        
    async def section(self, section_title: str):
        '''
        Get the plain text content of a section from `self.sections`.
        Returns None if `section_title` isn't found, otherwise returns a whitespace stripped string.

        This is a convenience method that wraps self.content.

        .. warning:: Calling `section` on a section that has subheadings will NOT return
            the full text of all of the subsections. It only gets the text between
            `section_title` and the next subheading, which is often empty.
        '''

        section = u"== {} ==".format(section_title)        
        content = await self.content
        
        try:
            index = content.index(section) + len(section)
        except ValueError:
            return None

        try:
            next_index = content.index("==", index)
        except ValueError:
            next_index = len(content)

        return content[index:next_index].lstrip("=").strip()
    
    @property
    async def content(self):
        '''
        Overwrites the `content` property that is called by the `section` property.
        This change enables the `content` property to be called independently in async.
        '''

        if not getattr(self, '_content', False):
            query_params = {
                'prop': 'extracts|revisions',
                'explaintext': '',
                'rvprop': 'ids'
            }
            if not getattr(self, 'title', None) is None:
                query_params['titles'] = self.title
            else:
                query_params['pageids'] = self.pageid
            request = await _async_wiki_request(query_params)
            self._content     = request['query']['pages'][self.pageid]['extract']
            self._revision_id = request['query']['pages'][self.pageid]['revisions'][0]['revid']
            self._parent_id   = request['query']['pages'][self.pageid]['revisions'][0]['parentid']

        return self._content
                