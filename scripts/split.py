import csv

def split():
    print("Splitting data into train, test and validation sets...")
    with open('data/fer2013.csv', 'r') as csvin:
        data = csv.reader(csvin)
        next(data)  # Skip the header

        train_data = [['emotion', 'pixels']]
        test_data = [['emotion', 'pixels']]
        validate_data = [['emotion', 'pixels']]

        for row in data:
            emotion_pixels = row[:2]  # Only take the emotion and pixels columns
            if row[-1] == 'Training':
                train_data.append(emotion_pixels)
            elif row[-1] == "PrivateTest":
                validate_data.append(emotion_pixels)
            elif row[-1] == "PublicTest":
                test_data.append(emotion_pixels)

        with open('data/train.csv', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train_data)

        with open('data/test.csv', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test_data)

        with open('data/validate.csv', 'w', newline='') as validate_file:
            writer = csv.writer(validate_file)
            writer.writerows(validate_data)
            
    print("Splitting completed successfully")
    
if __name__ == "__main__":
    split()