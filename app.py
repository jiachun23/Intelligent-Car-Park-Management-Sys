import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import PIL
from torchvision import transforms
import numpy as np
from labels import labels_dict
import easyocr
import datetime
import psycopg2
from datetime import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import calendar




def default_device():
    '''Indicate availablibity of GPU, otherwise return CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(tensorss, device):
    '''Move tensor to chosen device'''
    if isinstance(tensorss, (list, tuple)):
        return [to_device(x, device) for x in tensorss]
    return tensorss.to(device, non_blocking=True)


class DeviceDataloader():
    '''Wrap DataLoader to move the model to device'''

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        '''Yield batch of data after moving the data to a device'''
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        '''Return number of batches'''
        return len(self.dl)


# Check available device type
device = default_device()


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        image, label = batch
        out = self  # prediction generated
        loss = F.cross_entropy(out, label)  # Calculate loss using cross_entropy
        return loss

    def validation_step(self, batch):
        image, label = batch
        out = self  # predictioon generated
        loss = F.cross_entropy(out, label)  # Calculate loss using cross_entropy
        acc = accuracy(out, label)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, output):
        '''at the end of epoch, return average score (accuracy and cross entropy loss)'''
        batch_loss = [x['val_loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()

        batch_accs = [x['val_acc'] for x in output]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        '''Print out the score (accuracy and cross entropy loss) at the end of the epoch'''
        # result recorded using evaluate function
        print("Epoch [{}], train_loss: {:}, val_loss: {:}, val_acc: {:}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Resnet50(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 196)

    def forward(self, xb):
        return self.network(xb)


model = to_device(Resnet50(), device)

path = "car_model.pt"
car_model = torch.load(path, map_location=torch.device('cpu'))


def current_time():
    return datetime.datetime.now().replace(microsecond=0)


def findDay(date):
    day = datetime.datetime.strptime(date, '%Y %m %d').weekday()
    return (calendar.day_name[day])

def car_recogniser_entrance(our_img):
    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )
    # Setting auto commit false
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    car_image = our_img

    trans = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(inplace=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # preprocessing for prediction image
    input = trans(car_image)
    input = input.view(1, 3, 400, 400)

    output = car_model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())

    # return prediction label
    predicted_val = ([value for value in labels_dict.values()][prediction])
    st.text("Detected vehicle model: ")
    predicted_val

    # converting PIL object into numpy array for ocr
    new_array = np.array(car_image)
    reader = easyocr.Reader(['en'], gpu=False)
    bounds = reader.readtext(new_array, detail=0)

    st.text("Detected license plate number: ")
    num_plate = ' '.join([str(elem) for elem in bounds])
    num_plate

    enter_time = current_time()
    st.text("The vehicle enter the parking at:")
    enter_time
    time_enter = enter_time.strftime("%Y/%m/%d, %H:%M:%S")
    day_enter = enter_time.strftime("%Y %m %d")
    which_day = findDay(day_enter)



    sql = """INSERT INTO vehicle_data_entrance(vehicle_brand, plate_number, enter_time, week_day) VALUES(%s,%s,%s,%s)"""
    record_to_enter = (predicted_val, num_plate, time_enter,  which_day)
    cursor.execute(sql, record_to_enter)
    conn.commit()
    cursor.close()
    conn.close()


def car_recogniser_exit(our_img):
    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )
    # Setting auto commit false
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    car_image = our_img

    trans = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.RandomErasing(inplace=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # preprocessing for prediction image
    input = trans(car_image)
    input = input.view(1, 3, 400, 400)

    output = car_model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())

    # return prediction label
    predicted_val = ([value for value in labels_dict.values()][prediction])
    # comparing algo
    st.text("Detected vehicle model: ")
    predicted_val

    # converting PIL object into numpy array for ocr
    new_array = np.array(car_image)
    reader = easyocr.Reader(['en'], gpu=False)
    bounds = reader.readtext(new_array, detail=0)

    st.text("Detected license plate number: ")
    num_plate = ' '.join([str(elem) for elem in bounds])
    num_plate

    ext_time = current_time()
    st.text("The vehicle enter the parking at:")
    ext_time
    time_exit = ext_time.strftime("%Y/%m/%d, %H:%M:%S")
    day_exit = ext_time.strftime("%Y %m %d")
    which_day = findDay(day_exit)

    sql = """INSERT INTO vehicle_data_exit(vehicle_brand, plate_number, exit_time, week_day) VALUES(%s,%s,%s,%s)"""
    record_to_enter = (predicted_val, num_plate, time_exit, which_day)
    cursor.execute(sql, record_to_enter)
    conn.commit()
    cursor.close()
    conn.close()


def car_detection_entrance():
    global our_image
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
        </div>
        </body>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = PIL.Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        car_recogniser_entrance(our_image)


def car_detection_exit():
    global our_image
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
        </div>
        </body>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = PIL.Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        car_recogniser_exit(our_image)


# option at the side bar
options = st.sidebar.selectbox('Select an Option', ['Parking Entrance', 'Parking Exit', 'Parking Fee Calculation',
                                                    'Parking Database','No of Vehicles in the Parking','No of Parking Transactions by Day',
                                                    'Amount of Parking Fee Collected by Day'])

# title
st.set_option('deprecation.showfileUploaderEncoding', False)

if options == 'Parking Entrance':

    st.title("Car Model + License Plate Recognition for Parking Entrance")
    car_detection_entrance()


elif options == 'Parking Exit':

    st.title("Car Model + License Plate Recognition for Parking Exit")
    car_detection_exit()





elif options == 'Parking Fee Calculation':

    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )
    # Setting auto commit false
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    cursor1 = conn.cursor()

    st.title("Parking Fee Calculation")
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
        </div>
        </body>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    sql = """select vehicle_data_entrance.plate_number from vehicle_data_entrance order by vehicle_data_entrance.plate_number """

    cursor.execute(sql)

    result = [i[0] for i in cursor.fetchall()]


    dropdown = st.selectbox('Which vehicle you would like to choose?', (result))
    selection = dropdown
    st.text("Vehicle Entrance Time:")
    cursor.execute("select vehicle_data_entrance.enter_time from vehicle_data_entrance where vehicle_data_entrance.plate_number = %s",
        (selection,))
    result_enter = str(cursor.fetchone()[0])
    result_enter

    st.text("Vehicle Exit Time:")
    cursor.execute("select vehicle_data_exit.exit_time from vehicle_data_exit where vehicle_data_exit.plate_number = %s",
        (selection,))
    result_exit = str(cursor.fetchone()[0])
    if result_exit != None:
        result_exit
    else:
        st.text("The vehicle has not leave the parking yet.")

    # shows the calculation of parking duration and fee
    if result_enter and result_exit != None:
        st.text("Total parking duration (in minutes):")
        time_enter = dt.strptime(result_enter, "%Y/%m/%d, %H:%M:%S")
        time_exit = dt.strptime(result_exit, "%Y/%m/%d, %H:%M:%S")
        duration = (time_exit - time_enter) // datetime.timedelta(minutes=1)
        duration

    else:
        st.text("The vehicle is still in the parking.")

    st.header("Total Parking Fee:")
    if 0 <= duration <= 15:
        fee = 0.00
        st.subheader("Parking fee is free. No payment needed. :oncoming_automobile:")
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 15 <= duration <= 60:
        fee = 2.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)

    elif 60 <= duration <= 120:
        fee = 3.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)

    elif 120 <= duration <= 180:
        fee = 4.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 180 <= duration <= 240:
        fee = 5.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 240 <= duration <= 300:
        fee = 6.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 300 <= duration <= 360:
        fee = 7.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 360 <= duration <= 420:
        fee = 8.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)


    elif 420 <= duration <= 480:
        fee = 9.00
        st.subheader("Parking fee for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)

    else:
        fee = 10.00
        st.subheader("Parking for {} minutes is RM {:.2f} :oncoming_automobile:".format(duration, fee))
        sql = """UPDATE vehicle_data_exit SET duration = %s , fee = %s WHERE plate_number = %s """
        record_to_enter = (duration, fee, (selection,))
        cursor1.execute(sql, record_to_enter)



elif options == 'Parking Database':

    st.title("Parking Database")
    html_temp = """
           <body style="background-color:red;">
           <div style="background-color:teal ;padding:10px">
           <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
           </div>
           </body>
           """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )

    # Setting auto commit false
    conn.autocommit = True

    sql = pd.read_sql_query("""select vehicle_data_entrance.vehicle_brand, vehicle_data_entrance.plate_number,
                               vehicle_data_entrance.enter_time, vehicle_data_exit.exit_time,
                               vehicle_data_exit.duration, vehicle_data_exit.fee
                               from vehicle_data_entrance 
                               left join vehicle_data_exit ON 
                               vehicle_data_entrance.plate_number = vehicle_data_exit.plate_number 
			                   order by vehicle_data_entrance.plate_number """, conn)
    st.text("\n")
    st.text("\n")
    st.text("\n")
    df = pd.DataFrame(sql, columns=['vehicle_brand', 'plate_number', 'enter_time', 'exit_time','duration','fee'])
    df




elif options == 'No of Vehicles in the Parking':

    st.title("No of Vehicles in the Parking")
    html_temp = """
              <body style="background-color:red;">
              <div style="background-color:teal ;padding:10px">
              <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
              </div>
              </body>
              """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )

    # Setting auto commit false
    conn.autocommit = True
    cursor = conn.cursor()
    cursor1 = conn.cursor()

    sql = """select count(plate_number) from vehicle_data_entrance"""
    cursor.execute(sql)
    ent = cursor.fetchone()
    ent = int(''.join(map(str, ent)))

    sql_exit = """select count(plate_number) from vehicle_data_exit"""
    cursor1.execute(sql_exit)
    ext = cursor1.fetchone()
    ext = int(''.join(map(str, ext)))

    remain = ent-ext
    left = ext

    labels = ['Remaining in Car Park', 'Left the Car Park']
    values = [remain, left]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig)


elif options == 'No of Parking Transactions by Day':

    st.title("No of Parking Transactions by Day")
    html_temp = """
                 <body style="background-color:red;">
                 <div style="background-color:teal ;padding:10px">
                 <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
                 </div>
                 </body>
                 """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )

    # Setting auto commit false
    conn.autocommit = True
    cursor = conn.cursor()

    start_date = st.date_input('Starting Date: ')
    end_date = st.date_input('End Date: ')

    start_date = start_date.strftime("%Y/%m/%d")

    end_date = end_date.strftime("%Y/%m/%d")

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Monday' AND  enter_time >= %s AND  enter_time <= %s  """
    cursor.execute(sql,(start_date,end_date))
    mon = cursor.fetchone()

    mon = int(''.join(map(str, mon)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Tuesday' AND   enter_time >= %s AND  enter_time <= %s  """
    cursor.execute(sql, (start_date, end_date))
    tues = cursor.fetchone()
    tues = int(''.join(map(str, tues)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Wednesday' AND  enter_time >= %s AND  enter_time <= %s """
    cursor.execute(sql, (start_date, end_date))
    wed = cursor.fetchone()
    wed = int(''.join(map(str, wed)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Thursday' AND  enter_time >= %s AND  enter_time <= %s """
    cursor.execute(sql,(start_date,end_date))
    thurs = cursor.fetchone()
    thurs = int(''.join(map(str, thurs)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Friday' AND  enter_time >= %s AND  enter_time <= %s """
    cursor.execute(sql, (start_date, end_date))
    fri = cursor.fetchone()
    fri = int(''.join(map(str, fri)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Saturday' AND enter_time >= %s AND  enter_time <= %s """
    cursor.execute(sql, (start_date, end_date))
    sat = cursor.fetchone()
    sat = int(''.join(map(str, sat)))

    sql = """select count(week_day) from vehicle_data_entrance where week_day = 'Sunday' AND   enter_time >= %s AND  enter_time <= %s  """
    cursor.execute(sql,(start_date,end_date))
    sun = cursor.fetchone()
    sun = int(''.join(map(str, sun)))

    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekdays = [mon, tues, wed, thurs, fri, sat, sun]
    data = [go.Bar(
        x=labels,
        y=weekdays)]
    fig = go.Figure(data=data)
    fig.update_layout(title='No of Parking Transactions by Day', xaxis_title='Day',
                     yaxis_title='No of Parking Transactions')
    st.plotly_chart(fig)


elif options == 'Amount of Parking Fee Collected by Day':

    st.title("Amount of Parking Fee Collected by Day")
    html_temp = """
                 <body style="background-color:red;">
                 <div style="background-color:teal ;padding:10px">
                 <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
                 </div>
                 </body>
                 """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Establishing the connection
    conn = psycopg2.connect(
        database="vehicle", user='postgres', password='abc123', host='127.0.0.1', port='5432'
    )

    # Setting auto commit false
    conn.autocommit = True
    cursor = conn.cursor()

    start_date = st.date_input('Starting Date: ')
    end_date = st.date_input('End Date: ')

    start_date = start_date.strftime("%Y/%m/%d")
    end_date = end_date.strftime("%Y/%m/%d")

    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day = 'Monday' AND  exit_time >= %s AND  exit_time <= %s  """
    cursor.execute(sql,(start_date,end_date))
    mon = cursor.fetchone()[0]
    if mon != None:
        mon = mon

    else:
        mon = 0



    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day = 'Tuesday' AND   exit_time >= %s AND  exit_time <= %s  """
    cursor.execute(sql,(start_date,end_date))
    tues = cursor.fetchone()[0]
    if tues != None:
        tues = tues

    else:
        tues = 0


    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day  = 'Wednesday' AND  exit_time >= %s AND  exit_time <= %s """
    cursor.execute(sql, (start_date, end_date))
    wed = cursor.fetchone()[0]
    if wed != None:
        wed = wed

    else:
        wed = 0


    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day  = 'Thursday' AND  exit_time >= %s AND  exit_time <= %s """
    cursor.execute(sql,(start_date,end_date))
    thurs = cursor.fetchone()[0]
    if thurs != None:
        thurs = thurs

    else:
        thurs = 0


    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day = 'Friday' AND  exit_time >= %s AND  exit_time <= %s """
    cursor.execute(sql, (start_date, end_date))
    fri = cursor.fetchone()[0]
    if fri != None:
        fri = fri

    else:
        fri = 0

    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day = 'Saturday' AND exit_time >= %s AND exit_time <= %s """
    cursor.execute(sql,(start_date,end_date))
    sat = cursor.fetchone()[0]
    if sat != None:
        sat = sat

    else:
        sat = 0

    sql = """select cast(sum(fee) as int) from vehicle_data_exit where week_day = 'Sunday' AND  exit_time >= %s AND  exit_time <= %s  """
    cursor.execute(sql,(start_date,end_date))
    sun = cursor.fetchone()[0]
    if sun != None:
        sun = sun

    else:
        sun = 0


    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    amount = [mon,tues,wed,thurs,fri,sat,sun]
    data = [go.Scatter(x= labels,y=amount)]
    fig = go.Figure(data=data)
    fig.update_layout(title='Amount of Parking Fee Collected by Day',xaxis_title='Day',yaxis_title='Amount of Fee Collected (RM)')
    st.plotly_chart(fig)

else:
    st.text("The option is not exist. Please try again.")
