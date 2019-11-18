import os 
import re
import requests
import string
import pandas as pd
import urllib
from shutil import rmtree
from lxml import etree
from typing import List, Dict, Callable, Dict, Any

# Notes:
# --------
# TODO: self.verbose (UCIDatabaseEntry, UCIDatabase) is not used.
# TODO: complete the documentation
# TODO: add tests

class UCIDatabaseEntry():
    """ Simple structure to mangae single UCI's dataset """

    def __init__(self, name: str, url: str, data_types: List[str], default_tasks: List[str], attribute_types: List[str], no_instances: int, no_attributes: int, year: int, verbose: bool = True) -> None:
        """ Initialize default values of the class.
        
            Args:
                name (str): Name of the dataset
                url (str):  URL
                data_types (list): Determines types of input data (e.g. Multivariate, Time-Series, Sequential, ...)
                default_tasks (list): Tasks that could be resolved (e.g. Classification, Regression, ...)
                attribute_types (list): List of attribute types (e.g. Categorical, Integer, Real, ...)
                no_instances (int): self-explaintory
                no_attributes (int): self-explaintory
                year (int): self-explaintory
                verbose (bool): Messages will appear in the console if this flag is set to True
        """

        # Base parameters
        self.base_url = "https://archive.ics.uci.edu/ml/"
        self.verbose = verbose
        self.local_path = None
        
        # Save arguments
        self.name = name
        self.url = urllib.parse.urljoin(self.base_url, url)
        self.data_types = data_types
        self.default_tasks = default_tasks
        self.attribute_types = attribute_types
        self.no_instances = no_instances
        self.no_attributes = no_attributes
        self.year = year

    def download(self, location: str, overwrite: bool = False) -> None:
        """ Saves files from the dataset to disk 

            Args:
                location (str): path to directory where all datasets will be stored
                (note: this function will create a new folder with the same name as
                 dataset to save the files in it)
        """

        output_directory = os.path.join(location, self.name)
        self.local_path = output_directory

        # Skip if dataset is on the disk
        if os.path.isdir(output_directory) and not overwrite:
            return

        os.makedirs(output_directory, exist_ok=True)
        
        # Retrieve list of files of the dataset
        url_to_list_of_files = self.__get_download_url()
        file_list_as_html = requests.get(url_to_list_of_files).content
        files = etree.HTML(file_list_as_html).xpath(".//*[self::a]")
        
        # Save each file on the disk
        for single_file in files:
            if 'Parent Directory' not in single_file.text and 'Index' not in single_file.text:
                downloaded_file = requests.get(urllib.parse.urljoin(url_to_list_of_files, single_file.get('href')))
                open(os.path.join(output_directory, single_file.text.strip()), 'wb').write(downloaded_file.content)

    def __get_download_url(self) -> str:
        """ Retrieves URL to dataset's files """

        # XPath to "Data Folder" button on the dataset's page.
        datafolder_location_xpath = "//body/table[2]/tr/td/table[1]/tr/td[1]/p[1]/span[2]/a[1]"

        # Extracts the URL to list of files
        dataset_page = etree.HTML(requests.get(self.url).content)
        datafolder_location_path = dataset_page.xpath(datafolder_location_xpath)[0].get('href')
        return urllib.parse.urljoin(self.base_url, datafolder_location_path.replace('../', ''))
            
class UCIDatabase():
    """ Wrapper for UCI Datasets (archive.ics.uci.edu) """

    def __init__(self, url: str = "https://archive.ics.uci.edu/ml/datasets.php", output_directory: str = "datasets", cache_file: str = 'datasets.csv', load_from_cache: bool = True) -> None:

        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.url = url
        self.cache_file = cache_file
        self.datasets = []

        if not load_from_cache or not self.__load_cached_data():
           self.__generate_dataset()
           self.__cache_data()
           
    def get(self, function: Callable, download: bool = True) -> List[str]:
        """ Returns datasets that comply the filter function """

        filtered_datasets = list(filter(lambda ds: function(ds), self.datasets))

        if download:
            for dataset in filtered_datasets:
                dataset.download(self.output_directory)

        return filtered_datasets

    def __cache_data(self):
        """ Saves cached datasets as CSV file on the disk. """

        columns = ['name', 'url', 'data_types', 'default_tasks', 'attribute_types', 'no_instances', 'no_attribues', 'year']
        output_data = []

        for item in self.datasets:
            output_data.append([item.name, item.url, ','.join(item.data_types), ','.join(item.default_tasks), ','.join(item.attribute_types), item.no_instances, item.no_attributes, item.year])
            
        output_csv = pd.DataFrame(output_data, columns=columns)
        output_csv.to_csv(os.path.join(self.output_directory, self.cache_file), sep=';', index=False)

    def __load_cached_data(self) -> bool:
        """ Saves cached data as CSV on the disk.
        
            Returns:
                bool: True if CSV was found and loaded. False otherwise.
        """
        
        columns = ['name', 'url', 'data_types', 'default_tasks', 'attribute_types', 'no_instances', 'no_attribues', 'year']
        cache_file = os.path.join(self.output_directory, self.cache_file)

        if os.path.exists(cache_file):
            cache = pd.read_csv(cache_file, sep=';')

            for index, row in cache.iterrows():
                values = [row[item] for item in columns]
                preprocessed_values = [item.split(',') if type(item) is str and ',' in item else item for item in values]
                self.datasets.append(UCIDatabaseEntry(*preprocessed_values))

            return True
        return False
        
    def __generate_dataset(self, split_values_in_cols: List[int] = [1, 2, 3]) -> None:
        """ """

        def remove_non_ascii_characters(text: str) -> str:
            printable = set(string.printable)
            return ''.join(list(filter(lambda x: x in printable, text))).strip()
       
        # Create a dataset object
        self.datasets = []        

        # Define XPath that points to the table:
        table_xpath = "//body/table[2]/tr/td[2]/table[2]/*[self::tr]"

        # Retrieve the table (and remove the header)
        page = etree.HTML(requests.get(self.url).content)
        rows = page.xpath(table_xpath)[1:]

        # Generate dataset
        for row in rows:
            cells = row.xpath(".//p[@class='normal']")
            link = cells[0].xpath(".//a")[0]

            # Star gathering values
            values = [remove_non_ascii_characters(link.text), link.get('href')]
            for idx, cell in enumerate(cells[1:], start=1):

                # Remove non-printable characters
                cleaned_text = remove_non_ascii_characters(cell.text)

                # Split values into a list
                if idx in split_values_in_cols:
                    cleaned_text = list(map(lambda x: x.strip(), cleaned_text.split(',')))

                values.append(cleaned_text)

            self.datasets.append(UCIDatabaseEntry(*values))