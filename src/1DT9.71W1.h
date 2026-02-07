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
                        if (x[27] <= 0.007500000298023224) {
                            if (x[49] <= 0.0005000000237487257) {
                                if (x[28] <= -0.01699999999254942) {
                                    if (x[13] <= -0.4780000075697899) {
                                        if (x[32] <= 0.03800000064074993) {
                                            return 1;
                                        }

                                        else {
                                            return 2;
                                        }
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[14] <= -0.07349999994039536) {
                                        if (x[73] <= 0.001500000071246177) {
                                            return 4;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }

                                    else {
                                        return 4;
                                    }
                                }
                            }

                            else {
                                if (x[21] <= 0.20499999821186066) {
                                    if (x[32] <= 0.0924999974668026) {
                                        if (x[28] <= -0.017500000074505806) {
                                            return 3;
                                        }

                                        else {
                                            return 4;
                                        }
                                    }

                                    else {
                                        if (x[66] <= -0.03800000064074993) {
                                            return 1;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[3] <= 0.014500000048428774) {
                                        if (x[19] <= 0.49300000071525574) {
                                            return 2;
                                        }

                                        else {
                                            return 3;
                                        }
                                    }

                                    else {
                                        if (x[12] <= 1.0705000162124634) {
                                            return 3;
                                        }

                                        else {
                                            return 3;
                                        }
                                    }
                                }
                            }
                        }

                        else {
                            if (x[20] <= -0.9625000059604645) {
                                if (x[70] <= 0.15049999952316284) {
                                    if (x[68] <= 0.0005000000237487257) {
                                        if (x[70] <= 0.001500000071246177) {
                                            return 0;
                                        }

                                        else {
                                            return 2;
                                        }
                                    }

                                    else {
                                        if (x[12] <= 3.1234999895095825) {
                                            return 0;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[33] <= 0.11349999904632568) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                if (x[69] <= 9.89900016784668) {
                                    if (x[61] <= 0.0005000000237487257) {
                                        if (x[36] <= 0.009999999776482582) {
                                            return 3;
                                        }

                                        else {
                                            return 2;
                                        }
                                    }

                                    else {
                                        if (x[23] <= -0.009499999694526196) {
                                            return 2;
                                        }

                                        else {
                                            return 3;
                                        }
                                    }
                                }

                                else {
                                    if (x[61] <= 0.0005000000237487257) {
                                        if (x[37] <= 0.010999999940395355) {
                                            return 0;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }

                                    else {
                                        if (x[50] <= 0.0005000000237487257) {
                                            return 1;
                                        }

                                        else {
                                            return 1;
                                        }
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