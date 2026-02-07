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
                        if (x[27] <= 0.034374202601611614) {
                            if (x[0] <= -0.7526841461658478) {
                                if (x[12] <= 0.30113770067691803) {
                                    if (x[54] <= 1.5650760307908058) {
                                        return 4;
                                    }

                                    else {
                                        return 3;
                                    }
                                }

                                else {
                                    return 1;
                                }
                            }

                            else {
                                if (x[12] <= -1.260597288608551) {
                                    if (x[12] <= -1.277204990386963) {
                                        return 3;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[13] <= -1.4272929430007935) {
                                        return 1;
                                    }

                                    else {
                                        return 4;
                                    }
                                }
                            }
                        }

                        else {
                            if (x[31] <= 0.15674462169408798) {
                                if (x[13] <= 0.4259481579065323) {
                                    if (x[23] <= 0.018295538146048784) {
                                        return 0;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[15] <= 0.7075034379959106) {
                                        return 1;
                                    }

                                    else {
                                        return 2;
                                    }
                                }
                            }

                            else {
                                if (x[23] <= 0.018295538146048784) {
                                    if (x[18] <= 0.18115824460983276) {
                                        return 3;
                                    }

                                    else {
                                        return 2;
                                    }
                                }

                                else {
                                    if (x[13] <= 0.2574716992676258) {
                                        return 1;
                                    }

                                    else {
                                        return 3;
                                    }
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }