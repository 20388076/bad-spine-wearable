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
                        if (x[0] <= -4.324000000953674) {
                            if (x[2] <= 2.5500000715255737) {
                                if (x[1] <= -0.17599999904632568) {
                                    if (x[2] <= 1.5565000176429749) {
                                        return 8;
                                    }

                                    else {
                                        return 9;
                                    }
                                }

                                else {
                                    if (x[1] <= -0.17100000381469727) {
                                        return 8;
                                    }

                                    else {
                                        if (x[1] <= -0.159000001847744) {
                                            if (x[2] <= 1.301000028848648) {
                                                return 8;
                                            }

                                            else {
                                                return 9;
                                            }
                                        }

                                        else {
                                            return 8;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[0] <= -9.84749984741211) {
                                    return 8;
                                }

                                else {
                                    return 9;
                                }
                            }
                        }

                        else {
                            if (x[2] <= -4.615000009536743) {
                                if (x[5] <= -0.014500000048428774) {
                                    if (x[0] <= -0.9434999823570251) {
                                        if (x[5] <= -0.019499999471008778) {
                                            if (x[0] <= -1.1934999823570251) {
                                                return 0;
                                            }

                                            else {
                                                return 7;
                                            }
                                        }

                                        else {
                                            if (x[1] <= -0.03449999913573265) {
                                                return 3;
                                            }

                                            else {
                                                return 0;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[3] <= -0.004500000039115548) {
                                            if (x[0] <= 1.7879999876022339) {
                                                return 3;
                                            }

                                            else {
                                                return 1;
                                            }
                                        }

                                        else {
                                            if (x[1] <= -0.221500001847744) {
                                                return 3;
                                            }

                                            else {
                                                return 0;
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[0] <= -0.4815000146627426) {
                                        if (x[1] <= -6.109999895095825) {
                                            if (x[2] <= -7.0929999351501465) {
                                                return 6;
                                            }

                                            else {
                                                return 6;
                                            }
                                        }

                                        else {
                                            if (x[5] <= -0.005499999970197678) {
                                                return 5;
                                            }

                                            else {
                                                return 5;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[0] <= 0.17500000074505806) {
                                            if (x[0] <= -0.19950000196695328) {
                                                return 3;
                                            }

                                            else {
                                                return 4;
                                            }
                                        }

                                        else {
                                            if (x[4] <= -0.14199999580159783) {
                                                return 0;
                                            }

                                            else {
                                                return 2;
                                            }
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 2.174999952316284) {
                                    if (x[5] <= 0.009499999694526196) {
                                        if (x[0] <= 9.745500087738037) {
                                            return 11;
                                        }

                                        else {
                                            return 11;
                                        }
                                    }

                                    else {
                                        if (x[4] <= 0.04649999924004078) {
                                            return 10;
                                        }

                                        else {
                                            if (x[3] <= -0.007500000298023224) {
                                                return 11;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[5] <= -0.0020000000949949026) {
                                        return 11;
                                    }

                                    else {
                                        return 10;
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