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
                        if (x[2] <= 0.04800000041723251) {
                            return 3;
                        }

                        else {
                            if (x[12] <= 1.3440000414848328) {
                                if (x[41] <= 0.012500000186264515) {
                                    return 8;
                                }

                                else {
                                    if (x[51] <= 0.007500000298023224) {
                                        if (x[68] <= -0.04350000061094761) {
                                            return 11;
                                        }

                                        else {
                                            return 8;
                                        }
                                    }

                                    else {
                                        if (x[65] <= 3.876500129699707) {
                                            return 11;
                                        }

                                        else {
                                            return 8;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[16] <= -0.007999999914318323) {
                                    if (x[69] <= 933.5805206298828) {
                                        if (x[13] <= 0.005000000121071935) {
                                            if (x[51] <= 0.012500000128056854) {
                                                return 1;
                                            }

                                            else {
                                                return 12;
                                            }
                                        }

                                        else {
                                            if (x[51] <= 0.018499999307096004) {
                                                return 6;
                                            }

                                            else {
                                                if (x[17] <= -0.005000000121071935) {
                                                    return 9;
                                                }

                                                else {
                                                    return 6;
                                                }
                                            }
                                        }
                                    }

                                    else {
                                        if (x[73] <= 0.6319999769330025) {
                                            return 2;
                                        }

                                        else {
                                            return 14;
                                        }
                                    }
                                }

                                else {
                                    if (x[7] <= 0.0005000000237487257) {
                                        if (x[21] <= -0.646000012755394) {
                                            return 0;
                                        }

                                        else {
                                            return 5;
                                        }
                                    }

                                    else {
                                        if (x[3] <= 0.014500000048428774) {
                                            if (x[33] <= 0.11400000005960464) {
                                                return 10;
                                            }

                                            else {
                                                if (x[28] <= -0.02350000012665987) {
                                                    return 13;
                                                }

                                                else {
                                                    return 10;
                                                }
                                            }
                                        }

                                        else {
                                            if (x[24] <= -0.4855000078678131) {
                                                return 4;
                                            }

                                            else {
                                                return 7;
                                            }
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