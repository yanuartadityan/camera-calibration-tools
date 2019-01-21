import cv2 as cv
import argparse


def disp_video(id, filter=0):
    """this function will display video stream
    given input id"""

    captureHandler = cv.VideoCapture(id)

    # prepare all parameters for filtering
    if filter == 0:
        face_cascade = cv.CascadeClassifier('haar_frontal_adaboost.xml')

    # show video
    while(True):
        # retrieve frames from handler
        ret, frame = captureHandler.read()

        # convert color
        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # face filter
        if filter == 0:
            # detect face
            faces = face_cascade.detectMultiScale(grayed, 1.1, 5)

            # draw rectangular for the face
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # display
        cv.imshow('Color Video', frame)
        cv.imshow('Grayed Video', grayed)

        # exit on trigger
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release the video handler binding
    captureHandler.release()

    # close all window
    cv.destroyAllWindows()


if __name__ == '__main__':
    # parser video stream input
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', help='Pass the video capture ID, default = 0', default='0')
    args = parser.parse_args()

    disp_video(int(args.stream))
