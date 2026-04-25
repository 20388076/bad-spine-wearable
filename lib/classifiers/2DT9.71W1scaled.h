#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[69] <= 0.8948870301246643) {
                            if (x[14] <= 0.21880880743265152) {
                                return 1;
                            }

                            else {
                                if (x[17] <= -1.8553929328918457) {
                                    return 1;
                                }

                                else {
                                    return 2;
                                }
                            }
                        }

                        else {
                            return 0;
                        }
                    }

                protected:
                };
            }
        }
    }