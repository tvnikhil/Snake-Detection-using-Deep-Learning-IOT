import smtplib
import imghdr
from email.message import EmailMessage

Sender_Email = "akkalakarthik@gmail.com"
Password = "rutagksrnzehdlrs"


Reciever_Email = "tejakumarkalimera0107@gmail.com"

def sendmail(subject = "Snake DETECTED..." ):
    # return
    try:
        print("Sendimg Mail...", end = "")
        newMessage = EmailMessage()                         
        newMessage['Subject'] = subject
        newMessage['From'] = Sender_Email                   
        newMessage['To'] = Reciever_Email                   
        newMessage.set_content('Find the attached Image.') 

        with open("img.png", 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name

        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Sender_Email, Password)              
            smtp.send_message(newMessage)
        
        print(" Done...")
    except:
        print("Failed")
if __name__ == '__main__':
    sendmail()
        