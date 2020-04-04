import os 
import re
import requests
import string
import pandas as pd
import urllib
import shutil
from typing import List, Dict, Callable, Dict, Any

from shutil import rmtree
from lxml import etree

from . import preprocessing

# Disable unecessary wanings (e.g. disabled SSL verification)
requests.packages.urllib3.disable_warnings()


class UCIDatabaseEntry():
    """ Simple structure to mangae single UCI's dataset """

    def __init__(self, name: str, url: str, data_types: List[str], default_tasks: List[str], 
                attribute_types: List[str], no_instances: int, no_attributes: int, 
                year: int, output_directory: str, load_method: Callable = None, 
                verify_ssl: bool = True, verbose: bool = False) -> None:
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
        self.output_directory = output_directory
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
        self.verify_ssl = verify_ssl

        # Convertion method
        def raise_not_implemented_error(*args):
            raise NotImplementedError("The convertion method to DataFrame is not implemented!")

        self.load_method = raise_not_implemented_error if load_method is None else load_method

    def download(self, overwrite: bool = False) -> None:
        """ Saves files from the dataset to disk 

            Args:
                location (str): path to directory where all datasets will be stored
                (note: this function will create a new folder with the same name as
                 dataset to save the files in it)
        """

        self.local_path = os.path.join(self.output_directory, self.name)
        revert_changes = False

        try:
            
            # Skip if dataset is on the disk        
            if os.path.isdir(self.local_path) and not overwrite:
                if self.verbose:
                    print('The dataset exists on the disk.')
                return

            os.makedirs(self.local_path, exist_ok=True)
            
            # Retrieve list of files of the dataset
            dataset_directory = self._get_download_url()
            directory_index = requests.get(dataset_directory, verify=self.verify_ssl).content
            hyperlinks = etree.HTML(directory_index).xpath(".//*[self::a]")

            # Unnecessary URLs
            url_blacklist = set(['Parent Directory', 'Index', 'Name', 'Last modified', 'Size', 'Description'])

            # Save each file on the disk
            for a in hyperlinks:

                if a.text in url_blacklist:
                    continue

                # Retireve file url, name and content
                relative_url = a.get('href')
                url = urllib.parse.urljoin(dataset_directory, relative_url)
                name = urllib.parse.unquote(os.path.basename(relative_url))
                downloaded_file = requests.get(url, verify=self.verify_ssl)

                # Write content on the disk
                open(os.path.join(self.local_path, name), 'wb').write(downloaded_file.content)

        except Exception as exception:
            revert_changes = True
            raise exception

        finally:
            if revert_changes:
                shutil.rmtree(self.local_path)
                self.local_path = None


    def load(self) -> pd.DataFrame:
        """ Applies convertion method on downloaded files to create DataFrame """
        if self.local_path is None:
            raise Exception("Download the dataset first!")
        return self.load_method(self.local_path)

    def _get_download_url(self) -> str:
        """ Retrieves URL to dataset's files """

        # XPath to "Data Folder" button on the dataset's page.
        datafolder_location_xpath = "//body/table[2]/tr/td/table[1]/tr/td[1]/p[1]/span[2]/a[1]"

        # Extracts the URL to list of files
        dataset_page = etree.HTML(requests.get(self.url, verify=self.verify_ssl).content)
        datafolder_location_path = dataset_page.xpath(datafolder_location_xpath)[0].get('href')
        return urllib.parse.urljoin(self.base_url, datafolder_location_path.replace('../', ''))
    
    def __str__(self) -> str:
        return f'{self.name} Data Set ({self.local_path})'

    def __repr__(self) -> str:
        return self.__str__()
            

class UCIDatabase():
    """ Wrapper for UCI Datasets (archive.ics.uci.edu) """

    def __init__(self, url: str = "https://archive.ics.uci.edu/ml/datasets.php", 
                output_directory: str = "datasets", cache_file: str = 'datasets.csv', 
                load_from_cache: bool = True, verify_ssl: bool = True, 
                verbose: bool = False) -> None:

        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.url = url
        self.cache_file = cache_file
        self.datasets = []
        self.verbose = verbose
        self.verify_ssl = verify_ssl

        if not load_from_cache or not self._load_cached_data():          
            self._fetch_the_list_of_datasets()
            self._cache_data()
           
    def get(self, function: Callable, download: bool = True, first_only: bool = False) -> List[str]:
        """ Returns datasets that comply the filter function """
        output = []  
        filtered_datasets = list(filter(lambda ds: function(ds), self.datasets))
        _ = [dataset.download() for dataset in filtered_datasets if download]
        return filtered_datasets[0] if first_only else filtered_datasets

    def _cache_data(self):
        """ Saves cached datasets as CSV file on the disk. """

        columns = ['name', 'url', 'data_types', 'default_tasks', 'attribute_types', 'no_instances', 'no_attribues', 'year']
        output_data = []

        for item in self.datasets:
            output_data.append([item.name, item.url, '$'.join(item.data_types), '$'.join(item.default_tasks), '$'.join(item.attribute_types), item.no_instances, item.no_attributes, item.year])
            
        output_csv = pd.DataFrame(output_data, columns=columns)
        output_csv.to_csv(os.path.join(self.output_directory, self.cache_file), sep=';', index=False)

    def _load_cached_data(self) -> bool:
        """ Saves cached data as CSV on the disk.
        
            Returns:
                bool: True if CSV was found and loaded. False otherwise.
        """
        
        columns = ['name', 'url', 'data_types', 'default_tasks', 'attribute_types', 'no_instances', 'no_attribues', 'year']
        cache_file = os.path.join(self.output_directory, self.cache_file)

        if os.path.exists(cache_file):
            cache = pd.read_csv(cache_file, sep=';')

            for index, row in cache.iterrows():

                # Retrieve dataset descriptors
                values = [row[item] for item in columns]
                preprocessed_values = [item.split('$') if type(item) is str and '$' in item else item for item in values]
                preprocessed_values.append(self.output_directory)
                preprocessed_values.append(self._predefined_load_methods(preprocessed_values[0]))
                dataset_entry = UCIDatabaseEntry(*preprocessed_values)

                # Set parameters
                dataset_entry.verbose = self.verbose
                dataset_entry.verify_ssl = self.verify_ssl

                self.datasets.append(dataset_entry)

            return True
        return False
        
    def _fetch_the_list_of_datasets(self, split_values_in_cols: List[int] = [1, 2, 3]) -> None:
        """ Extracts list of available UCI dataset from the website """

        def remove_non_ascii_characters(text: str) -> str:
            printable = set(string.printable)
            return ''.join(list(filter(lambda x: x in printable, text))).strip()
       
        # Create a dataset object
        self.datasets = []        

        # Define XPath that points to the table:
        table_xpath = "//body/table[2]/tr/td[2]/table[2]/*[self::tr]"

        # Retrieve the table (and remove the header)
        page = etree.HTML(requests.get(self.url, verify=self.verify_ssl).content)
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

            values.append(self.output_directory)
            values.append(self._predefined_load_methods(values[0]))
            self.datasets.append(UCIDatabaseEntry(*values))

    def _predefined_load_methods(self, dataset_name) -> None:
        """ Returns function that converts downloaded dataset to DataFrame. """

        # List of predefined methods
        predefined_methods = {
            'Phishing Websites': preprocessing._phishing_websites,
            'Breast Cancer Wisconsin (Diagnostic)': preprocessing._breast_cancer_wisconsin_diag,
            'Bank Marketing': preprocessing._bank_marketing,
            'Adult': preprocessing._adult,
            'Skin Segmentation': preprocessing._skin_segmentation

        }

        return predefined_methods[dataset_name] if dataset_name in predefined_methods.keys() else None 
