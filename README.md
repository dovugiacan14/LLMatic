# Data Normalization

This repository is designed to normalize data include: remove duplicates, split complex data, date conversation, currency unification, and extract information from Nayose API.

## Requirements
 - Python 3.10+(recommend Python 3.11)
 - pip 24.0

## Installation

To setup environment and install the project dependencies run:

1. Create a virtual environment:
   ```sh
   python -m venv .venv
   ```

2. Activate the virtual enviroment:
   On Windows:
   ```sh
   .\.venv\Scripts\activate
   ```

3. Upgrade pip to the latest version
   To update pip to the newest version, use the following command:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Check the pip version:
   ```bash
   pip --version
   ```

5. Check the Python version:
   ```bash
   python --version
   ```

6. Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

7. Change NAYOSE API information if neccessary at ```.env``` file.

## Usage

Run the script with the following command:

```bash
python -m src.main -i ./path_to_input_file -c ./path_to_config
```

Or you can use **--input_file** rather than **-i** and **--config_file** rather than **-c**.
```bash
python -m src.main --input_file ./path_to_input_file --config_file ./path_to_config
```

## Command line arguments

 Below is the example of the command line arguments that can be used with the script.

- `--help`
   - Type: `boolean`
   - Optional: `True`
   - Description: Show the help message and exit

## Example usage

```bash
python -m src.main -i data_test_navi_virtual.xlsx -c config/navi_config.yaml
```
The result is stored at: ```data/proceseed/{time_now}/result_normalized_only_{time_now}```

## Enviroment Variables
You can adjust the environment variables in the .env file. The information includes:



- **NAYOSE_HOST**: The host address of the Nayose API.

- **REQUEST_TOKENS_ENDPOINT**: The endpoint of the Nayose API used to obtain the token. Example `api/auth/access/`,

- **SEARCH_ENDPOINT**: The endpoint of the Nayose API used for search functionality. Example `api/search/`

- **USER_NAME**: Invalid username used to retrieve the API token.

- **PASSWORD**: Corresponding password.

- **LIMIT**: Number of records are sent to Nayose API search at once. (MAX = 500)

- **MAX_RETRIES**: The number of retry attempts when retrieving the API token.

**Please read NayoseAPI Document for more detail.**

## How to Use Parameters in the Configuration File

### 1. **Remove Duplication**

| Parameters | Description |
| --- | --- |
| `dedup_columns` | field to specify the columns used for handling duplicates. |
| `Rule1` | contains list of columns that if the values of any two rows in these columns are identical, the rows are considered duplicates. |
| `Rule2` | if two or more rows have the same value in this column, they are considered duplicates. (usually ID column). |
| `order_by*` | usually a date column to identify the newest information row |

**Example**

![Alt text](assets/remove_duplication_example.png)
----------------------------------------

###  2. **Nayose API**

| Parameters | Description |
| --- | --- |
| `nayose` | field to specify the columns used for finding ```company name``` and ```company number``` from Nayose API. |
| `search_by` | contains columns (```column_name```) and corresponding attributes (```api_property```), used for preparing the request to send information to the Nayose API. |
| `set_to` | contains columns (```column_name```) and corresponding attributes (```api_property```), used for exporting information to CSV file after having response from Nayose API. |

**Example**

![Alt text](assets/nayose_example.png)

###  3. **Other Parameters**

-  **Normalize**

| Parameters | Description |
| --- | --- |
| `normalize` | normalize columns to define types. |
| `type` | type of task |
|  | ```split```:  columns to split data (**task:** Splitting Cells with Multiple Data Entries). |
|  | ```date```:  datetime column (**task:** Date Conversion). |
|  | ```currency```:  currency column (**task:** Currency Unification). |
| `split_by_regex` | put the regex to split cells data. |
| `skip_if_contains:` | specify keywords; if any word contains one of these keywords, the word will be skipped and not processed.  |

**Example**

![Alt text](assets/normalize_example.png)

- **Create Columns**

| Parameters | Description |
| --- | --- |
| `create_columns` | create columns from existing column. |
| `from_column` | column name. |
| `type` | type of task |
|  | ```staff_numbers```:  columns need to extract numbers of staff for non-consolidated. |
|  | ```katakana```:  the hiragana column, which needs to be converted to katakana characters. |

**Example**

![Alt text](assets/create_column_example.png)