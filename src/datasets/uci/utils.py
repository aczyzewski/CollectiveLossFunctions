from typing import Tuple, List, Dict, Any, Callable


def iterfile(filename: str, ignore_empty_lines: bool = True,
             filters: List[Callable[[str], bool]] = []) -> Any:
    """ Yields non-empty rows of a file """

    if ignore_empty_lines:
        filters.append(lambda x: len(x) == 0)

    for row in open(filename, 'r').read().split('\n'):
        if not any([func(row) for func in filters]):
            yield row


def load_arff_file(location: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """ Loads .arff file from disk and extracts metadata + training examples """

    # TODO: Consider more types of parameters (@...)

    metadata = {}
    examples, columns = [], []
    data_section = False

    for row in iterfile(location):

        if row.startswith('@data'):
            data_section = True
            continue

        if data_section:
            examples.append(dict(zip(columns, row.split(','))))
        else:
            if row.startswith('@attribute'):
                columns.append(row.split(' ')[1].lower())

    metadata['columns'] = columns
    metadata['no_examples'] = len(examples)
    return examples, metadata
