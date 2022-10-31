#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>

class Point {
public:
    int x;
    int y;
    Point(int x, int y) : x(x), y(y) {}
};

static float max_f32(cv::Mat img) {
    float max = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float color = img.at<float>(i, j);
            if (color > max) max = color;
        }
        if (max == UINT8_MAX) return max;
    }
    return max;
}

class BallDetector {
public:
    cv::Ptr<cv::SimpleBlobDetector> blob_detector;
    BallDetector() {
        cv::SimpleBlobDetector::Params ball_detector_params;
        ball_detector_params.minThreshold = 10;
        ball_detector_params.maxThreshold = 200;
        ball_detector_params.filterByArea = true;
        ball_detector_params.minCircularity = 0.5f;
        ball_detector_params.filterByConvexity = false;
        ball_detector_params.minConvexity = 0.5f;
        ball_detector_params.filterByInertia = false;
        ball_detector_params.minInertiaRatio = 0.5f;
        blob_detector = cv::SimpleBlobDetector::create(ball_detector_params);
    }
    std::vector<Point>* detect(cv::Mat img) {
        cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat gray_img;
        cv::Mat float_gray_img(gray_img.rows, gray_img.cols, CV_32F);
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);


        gray_img.convertTo(float_gray_img, CV_32F);

        cv::Mat dst;
        cv::Mat dst1;

        cv::cornerHarris(float_gray_img, dst, 2, 3, 0.04);
        cv::morphologyEx(dst, dst1, cv::MORPH_CLOSE, kernel_close);
        cv::morphologyEx(dst1, dst, cv::MORPH_CLOSE, kernel_close);

        cv::Mat blank_img = cv::Mat::zeros(dst.rows, dst.cols, CV_8U);

        float dst_max = max_f32(dst);

        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                float dst_color = dst.at<float>(i, j);
                if (dst_color > dst_max * 0.01f) blank_img.at<uint8_t>(i, j) = 255;
                else blank_img.at<uint8_t>(i, j) = 0;
            }
        }

        cv::Mat blured_blank_img(blank_img.rows, blank_img.cols, CV_8U);

        cv::medianBlur(blank_img, blured_blank_img, 5);

        int rows = blured_blank_img.rows;

        std::vector<cv::Vec3f> circles;

        cv::HoughCircles(blured_blank_img, circles, cv::HOUGH_GRADIENT, 1, rows / 8, 100, 1, 1, 4);

        auto rez = new std::vector<Point>();

        for (int i = 0; i < circles.size(); i++) {
            cv::Vec3i circle = circles[i];

            int delta = circle[2] * 2;

            cv::Mat ball_img = img(cv::Range(circle[1] - delta, circle[1] + delta), cv::Range(circle[0] - delta, circle[0] + delta));

            cv::Mat resized_ball_img;

            cv::resize(ball_img, resized_ball_img, cv::Size(48, 48));

            std::vector<cv::KeyPoint> ball_keypoints;
   
            blob_detector->detect(resized_ball_img, ball_keypoints);
            if (ball_keypoints.size() > 0) {
                rez->push_back(Point(circle[0], circle[1]));
            }
        }

        return rez;
    }

    ~BallDetector() {
        blob_detector.release();
    }
};

int main() {
    BallDetector detector;
    cv::Mat img = cv::imread("img.png");
    auto rez = detector.detect(img);
    std::cout << rez->size() << std::endl;
    for (auto& p : *rez) {
        std::cout << p.x << ", " << p.y << std::endl;
    }
    delete rez;
}