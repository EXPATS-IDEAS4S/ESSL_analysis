import logging
from botocore.exceptions import ClientError



def _print_event_day_summary(df, label):
    """
    Print a concise summary showing total events and number of unique days.
    """
    total_events = len(df)
    unique_days = df['day_id'].nunique() if 'day_id' in df.columns else 0
    print(f"{label}: {total_events} events across {unique_days} unique days")




def read_file(s3, file_name, bucket):
    """Upload a file to an S3 bucket
    :param s3: Initialized S3 client object
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :return: object if file was uploaded, else False
    """
    try:
        #with open(file_name, "rb") as f:
        obj = s3.get_object(Bucket=bucket, Key=file_name)
        #print(obj)
        myObject = obj['Body'].read()
    except ClientError as e:
        logging.error(e)
        return None
    return myObject



def list_bucket_objects_once(s3_client, bucket, prefix=None, verbose=False):
    """
    List objects in an S3 bucket with a single non-paginated request.

    Returns the list found under the 'Contents' key (may be empty).
    """
    params = {'Bucket': bucket}
    if prefix:
        params['Prefix'] = prefix

    try:
        response = s3_client.list_objects(**params)
    except ClientError as e:
        logging.error("Error listing objects: %s", e)
        return []

    contents = response.get('Contents', [])
    if verbose and contents:
        for obj in contents:
            print(obj.get('Key'))
    return contents


def list_all_bucket_objects(s3_client, bucket, prefix=None, verbose=True):
    """
    Return a list with all objects in the bucket using the paginator.

    Each entry is the dict returned by S3 (contains 'Key', 'LastModified', etc.).
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pagination_params = {'Bucket': bucket}
    if prefix:
        pagination_params['Prefix'] = prefix

    all_objects = []
    try:
        pages = paginator.paginate(**pagination_params)
        for page in pages:
            contents = page.get('Contents', [])
            if verbose and contents:
                print(f"Got {len(contents)} objects in page")
            all_objects.extend(contents)
    except ClientError as e:
        logging.error("Error during pagination: %s", e)
        return []

    if verbose:
        print(f"Total objects in bucket '{bucket}': {len(all_objects)}")
    return all_objects


