# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import imutils
import cv2


# Measured width of box in inches
width = 1.49606
width_cm = round(width / 0.39370079, 2)

# debug info
debugging = False
debug_pos = 75

# Colors for lines on distance vectors
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))

# Color mask boundaries (BLUE)
lower_bound = [86, 31, 4]
upper_bound = [220, 88, 50]

# Video capture source
cap = cv2.VideoCapture(0)


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# Returns a masked version of an image based on RGB boundaries
def color_masked_image(input_image):

    # create NumPy arrays from the boundaries
    lower = np.array(lower_bound, dtype="uint8")
    upper = np.array(upper_bound, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(input_image, lower, upper)
    output_image = cv2.bitwise_and(input_image, input_image, mask=mask)
    return output_image


# Performs normal canning edge manipulation
def canny_edge_preparation(input_image):

    # convert image to grayscale
    _gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # blur it slightly
    _gray = cv2.GaussianBlur(_gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    _edged = cv2.Canny(_gray, 50, 100)
    _edged = cv2.dilate(_edged, None, iterations=1)
    _edged = cv2.erode(_edged, None, iterations=1)
    return _edged


def find_contours_in_edge_map(input_image):
    # find contours in the edge map
    _cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _cnts = _cnts[0] if imutils.is_cv2() else _cnts[1]
    return _cnts


def compute_bounding_box(input_contour):
    # compute the rotated bounding box of the contour
    _box = cv2.minAreaRect(input_contour)
    _box = cv2.cv.BoxPoints(_box) if imutils.is_cv2() else cv2.boxPoints(_box)
    _box = np.array(_box, dtype="int")
    return _box


def compute_bounding_box_for_contour(input_contour):
    # compute the rotated bounding box of the contour
    _box = compute_bounding_box(input_contour)

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    _box = perspective.order_points(_box)

    # compute the center of the bounding box
    _cX = np.average(_box[:, 0])
    _cY = np.average(_box[:, 1])
    return _box, _cX, _cY


def compute_and_add_debug_info(img, input_ref_obj):
    input_image = img

    cv2.rectangle(input_image, (0, 0), (150, 50), lower_bound, -1)
    cv2.rectangle(input_image, (150, 0), (300, 50), upper_bound, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(input_image, 'rgb_lower',
                (15, 30), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(input_image, 'rgb_upper',
                (165, 30), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(input_image, 'detected contours: ' + str(len(cnts)),
                (15, debug_pos), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if refObj is not None:
        cv2.putText(orig, 'reference coords: x:' + str(input_ref_obj[1][0]) + ' y:' + str(input_ref_obj[1][1]),
                    (15, debug_pos + 25), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(orig, 'reference coords: ',
                    (15, debug_pos + 25), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(input_image, 'ref object width (inch): ' + str(width),
                (15, debug_pos + 50), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(input_image, 'ref object width (cm): ' + str(width_cm),
                (15, debug_pos + 75), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return input_image


def find_reference_object_pos(input_image):
    masked = color_masked_image(input_image)
    canny_masked = canny_edge_preparation(masked)
    cnt_canny_masked = find_contours_in_edge_map(canny_masked)

    # Find the largest contour
    largest_contour = None
    for _c in cnt_canny_masked:
        if largest_contour is None:
            largest_contour = _c
        elif cv2.contourArea(_c) > cv2.contourArea(largest_contour):
            largest_contour = _c

    if largest_contour is not None and cv2.contourArea(largest_contour) > 200:
        # compute the rotated bounding box of the contour
        _box, _cX, _cY = compute_bounding_box_for_contour(largest_contour)

        # unpack the ordered bounding box, then compute the
        # midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and
        # bottom-right
        (_tl, _tr, _br, _bl) = _box
        (_tlblX, _tlblY) = midpoint(_tl, _bl)
        (_trbrX, _trbrY) = midpoint(_tr, _br)

        # compute the Euclidean distance between the midpoints,
        # then construct the reference object
        _D = dist.euclidean((_tlblX, _tlblY), (_trbrX, _trbrY))
        _refObj = (_box, (_cX, _cY), _D / width)
        return _refObj


while True:
    # Capture frame-by-frame  
    ret, image = cap.read()

    # Make a copy of the image to draw on
    orig = image.copy()

    # Perform canning edge manipulations
    edged = canny_edge_preparation(image)

    # find contours in the edge map
    cnts = find_contours_in_edge_map(edged)

    # get the reference object
    refObj = find_reference_object_pos(image)

    if refObj is not None:

        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 200:
                continue

            # Compute the rotated bounding box of the contour
            box, cX, cY = compute_bounding_box_for_contour(c)

            # draw the contours on the image
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 0, 255), 2)
            cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

            # stack the reference coordinates and the object coordinates
            # to include the object center
            refCoords = np.vstack([refObj[0], refObj[1]])
            objCoords = np.vstack([box, (cX, cY)])

            # loop over the original points
            for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):

                # Now this is a hack.
                if color == (255, 0, 255):
                    # draw circles corresponding to the current points and
                    # connect them with a line
                    cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
                    cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
                    cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
                             color, 2)

                    # compute the Euclidean distance between the coordinates,
                    # and then convert the distance in pixels to distance in
                    # units
                    D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
                    (mX, mY) = midpoint((xA, yA), (xB, yB))
                    cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if debugging:
        final_image = compute_and_add_debug_info(orig, refObj)
    else:
        final_image = orig

    # Display the resulting frame
    cv2.imshow('Distance Mapping', final_image)

    # Key press handler
    key = cv2.waitKey(1) & 0xFF

    # If Q is pressed, quit
    if key == ord('q'):
        break
    # If D is pressed, debug info
    elif key == ord('d'):
        debugging = not debugging

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
