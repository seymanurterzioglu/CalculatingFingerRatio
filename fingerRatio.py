from cv2 import cv2
import mediapipe as mp
import numpy as np

number = 1  # for name the photo

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# photo taken with camera

cap = cv2.VideoCapture(0)  ### VideoCapture(1)= usb camera


##  Functions  ##

# locations of landmarks
def location(handlms, id):
    h, w, c = image.shape
    # calculate locations
    cx = int(handlms.landmark[id].x * w)
    cy = int(handlms.landmark[id].y * h)

    return (cx, cy)


def print_location(image, results):
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            # print("id= ", id, ", lm= ", lm)
            h, w, c = image.shape
            # calculate locations
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            print(id, "\t cx: ", cx, "\t cy: ", cy)


def get_handedness(index, results):
    output = None  # free
    for handLms in results.multi_hand_landmarks:
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:
                label = classification.classification[0].label

                coords = location(handLms, index)
                output = label, coords

    return output


# the index that came from joint_list will give angles numbers of finger
def calculate_finger_size(image, hand, joint_list):  # for 1.finger [4,3,2]  to down from up
    for hand in results.multi_hand_landmarks:
        distance = 0
        first = np.array([hand.landmark[joint_list[0]].x, hand.landmark[joint_list[0]].y])  # first coordinates
        second = np.array([hand.landmark[joint_list[1]].x, hand.landmark[joint_list[1]].y])  # second coordinates
        third = np.array([hand.landmark[joint_list[2]].x, hand.landmark[joint_list[2]].y])  # third coordinates

        # np.array([ , ]) must be like that
        x_y1 = np.array(
            [((first[0] - second[0]) * (first[0] - second[0])), ((first[1] - second[1]) * (first[1] - second[1]))])
        total1 = x_y1[0] + x_y1[1]
        distance1 = np.sqrt(total1)

        distance += distance1

        x_y2 = np.array(
            [((second[0] - third[0]) * (second[0] - third[0])), ((second[1] - third[1]) * (second[1] - third[1]))])
        total2 = x_y2[0] + x_y2[1]
        distance2 = np.sqrt(total2)

        distance += distance2

        if len(joint_list) >= 4:
            fourth = np.array([hand.landmark[joint_list[3]].x, hand.landmark[joint_list[3]].y])  # fourth coordinates

            x_y3 = np.array(
                [((fourth[0] - third[0]) * (fourth[0] - third[0])), ((fourth[1] - third[1]) * (fourth[1] - third[1]))])
            total3 = x_y3[0] + x_y3[1]
            distance3 = np.sqrt(total3)

            distance += distance3
        else:
            continue  # continue can be used in for or while loop

        # if you want
        '''
        cv2.putText(image, str(distance), tuple(np.multiply(first, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 1, cv2.LINE_AA)
                    
        '''

    return image, distance


while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)

    cv2.imshow("Camera", image)

    # if there is a photo with that name
    if cv2.waitKey(1) & 0xFF == ord('c'):  # click 'c' twice
        if cv2.imread('images/pic' + str(number) + '.jpg') is None:
            cv2.imwrite('images/pic' + str(number) + '.jpg', image)
            original = cv2.imread('images/pic' + str(number) + '.jpg')
        else:
            number += 1
            cv2.imwrite('images/pic' + str(number) + '.jpg', image)
            original = cv2.imread('images/pic' + str(number) + '.jpg')
        break

# static photo

# original = cv2.imread('images/right4.jpg')  ###images/...   dont forget(if there is a images file)

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate('images'):

        if original is None:
            print('None')
            break
        image = original.copy()

        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and image height,width
        if not results.multi_hand_landmarks:
            continue

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):

                    # draw circle to landmarks
                    for i in range(2, 20):
                        if id >= 2:
                            cv2.circle(image, location(handLms, id), 3, (255, 255, 255), cv2.FILLED)

                    # 1.finger landmark connections
                    cv2.line(image, location(handLms, 2), location(handLms, 3), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 3), location(handLms, 4), (0, 0, 0), 1)

                    # 1.finger size
                    joint1 = [4, 3, 2]  # joint_list[0]= 1.finger landmarks id
                    image, finger1size = calculate_finger_size(image, results,
                                                               joint1)

                    # 2.finger landmark connections
                    cv2.line(image, location(handLms, 5), location(handLms, 6), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 6), location(handLms, 7), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 7), location(handLms, 8), (0, 0, 0), 1)

                    # 2.finger size
                    joint2 = [8, 7, 6, 5]  # joint_list[1]= 2.finger landmarks id
                    image, finger2size = calculate_finger_size(image, results,
                                                               joint2)

                    # 3.finger landmark connections
                    cv2.line(image, location(handLms, 9), location(handLms, 10), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 10), location(handLms, 11), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 11), location(handLms, 12), (0, 0, 0), 1)

                    # 3.finger size
                    joint3 = [12, 11, 10, 9]  # joint_list[3]= 4.finger landmarks id
                    image, finger3size = calculate_finger_size(image, results,
                                                               joint3)

                    # 4.finger landmark connections
                    cv2.line(image, location(handLms, 13), location(handLms, 14), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 14), location(handLms, 15), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 15), location(handLms, 16), (0, 0, 0), 1)

                    # 4.finger size
                    joint4 = [16, 15, 14, 13]  # joint_list[3]= 4.finger landmarks id
                    image, finger4size = calculate_finger_size(image, results,
                                                               joint4)

                    # 5.finger landmark connections
                    cv2.line(image, location(handLms, 17), location(handLms, 18), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 18), location(handLms, 19), (0, 0, 0), 1)
                    cv2.line(image, location(handLms, 19), location(handLms, 20), (0, 0, 0), 1)

                    # 5.finger size
                    joint5 = [20, 19, 18, 17]  # 5.finger landmark id
                    image, finger5size = calculate_finger_size(image, results,
                                                               joint5)

                    # get handedness
                    if get_handedness(id, results):
                        label, coord = get_handedness(id, results)
                        cv2.putText(image, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                connections = original.copy()

                # draw hand landmarks connections on the photo
                mp_drawing.draw_landmarks(connections, handLms, mp_hands.HAND_CONNECTIONS)

# original photo, digit photo , mediapipe landmark photo
all_image = np.hstack((original, image))  # if you want mediapipe landmark pic add connections
cv2.imshow('All Images', all_image)
cv2.imwrite('images/Land' + str(number) + '.jpg', all_image)

# which hand? and hand.landmarks location
print('Handedness:', results.multi_handedness)
print_location(image, results)  # this is working correctly

# fingers digit ratio
print('1D=  ' + str(round(finger1size, 3)))
print('2D=  ' + str(finger2size))
print('3D=  ' + str(finger3size))
print('4D=  ' + str(finger4size))
print('5D=  ' + str(finger5size))
print('2D:3D= ' + str((finger2size / finger3size)))
print('The Most Reliable---2D:4D= ' + str((finger2size / finger4size)))
print('2D:5D= ' + str((finger2size / finger5size)))
print('3D:4D= ' + str((finger3size / finger4size)))
print('3D:5D= ' + str((finger3size / finger5size)))
print('4D:5D= ' + str((finger4size / finger5size)))

cv2.waitKey(0)
cv2.destroyAllWindows()
