#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class RandomForest {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        uint8_t votes[5] = { 0 };
                        // tree #1
                        if (x[2] <= 0.15464795380830765) {
                            if (x[28] <= 0.6078092753887177) {
                                if (x[74] <= 0.1867204997688532) {
                                    if (x[28] <= 0.45590999722480774) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[0] += 1;
                                    }
                                }

                                else {
                                    votes[0] += 1;
                                }
                            }

                            else {
                                votes[4] += 1;
                            }
                        }

                        else {
                            if (x[0] <= -0.8584603071212769) {
                                votes[3] += 1;
                            }

                            else {
                                if (x[36] <= -0.11090455204248428) {
                                    if (x[29] <= 0.03281394764780998) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[11] <= 0.13438698649406433) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[3] += 1;
                                    }
                                }
                            }
                        }

                        // tree #2
                        if (x[26] <= -0.08028349280357361) {
                            votes[4] += 1;
                        }

                        else {
                            if (x[34] <= -0.4116825759410858) {
                                if (x[25] <= 0.0001462213695049286) {
                                    votes[0] += 1;
                                }

                                else {
                                    if (x[20] <= 1.2665987610816956) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                votes[3] += 1;
                            }
                        }

                        // tree #3
                        if (x[28] <= 0.6078092753887177) {
                            if (x[31] <= 0.15674462169408798) {
                                if (x[25] <= -0.5973592102527618) {
                                    votes[4] += 1;
                                }

                                else {
                                    if (x[20] <= 1.1899933218955994) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }
                            }

                            else {
                                votes[3] += 1;
                            }
                        }

                        else {
                            votes[4] += 1;
                        }

                        // tree #4
                        if (x[23] <= -0.06525728851556778) {
                            votes[4] += 1;
                        }

                        else {
                            if (x[21] <= -0.6987386047840118) {
                                if (x[20] <= -0.7086697518825531) {
                                    if (x[71] <= 0.7366297841072083) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[22] <= 0.12340962886810303) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[32] <= -0.2783697098493576) {
                                    if (x[16] <= -0.4056455045938492) {
                                        votes[4] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[27] <= 0.06366218253970146) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[3] += 1;
                                    }
                                }
                            }
                        }

                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 5; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                protected:
                };
            }
        }
    }