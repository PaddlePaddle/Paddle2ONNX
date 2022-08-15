//
// Created by zhengbicheng on 2022/8/11.
//

#ifndef BASE_CONFIG_H
#define BASE_CONFIG_H

typedef struct rectangle {
    float x0;
    float y0;
    float x1;
    float y1;
} rectangle;

typedef struct Bbox {
    rectangle rec;
    float score;
} Bbox;

typedef struct BboxWithID {
    Bbox box;
    int id;
} BboxWithID;
#endif //BASE_CONFIG_H
