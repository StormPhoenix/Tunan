//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/utils/image_utils.h>
#include <tunan/utils/file_utils.h>

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION

#include <ext/stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <ext/stb/stb_image_write.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        static float *stbReadImage(const std::string &filename, int *width, int *height,
                                   int desire_channels, int *channels_in_file = nullptr) {
            int channelsInFile;
            unsigned char *image = stbi_load(filename.c_str(), width, height, &channelsInFile, 0);
            if (image == nullptr) {
                return nullptr;
            }

            int w = *width;
            int h = *height;
            float *ret = (float *) malloc(sizeof(float) * w * h * desire_channels);

            double weight = 1.0 / 255.0;
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    int imageOffset = (((h) - (row + 1)) * w + col) * channelsInFile;
                    int spectrumOffset = (row * w + col) * desire_channels;
                    for (int ch = 0; ch < channelsInFile && ch < desire_channels; ch++) {
                        unsigned char val = image[imageOffset + ch];
                        ret[spectrumOffset + ch] = val * weight;

                    }
                }
            }

            stbi_image_free(image);
            if (channels_in_file != nullptr) {
                (*channels_in_file) = channelsInFile;
            }
            return ret;
        }

        namespace pfm {
            static constexpr bool hostLittleEndian =
#if defined(__BYTE_ORDER__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                    true
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            false
#else
#error "__BYTE_ORDER__ defined but has unexpected value"
#endif
#else
#if defined(__LITTLE_ENDIAN__) || defined(__i386__) || defined(__x86_64__) || \
      defined(_WIN32) || defined(WIN32)
                    true
#elif defined(__BIG_ENDIAN__)
            false
#elif defined(__sparc) || defined(__sparc__)
            false
#else
#error "Can't detect machine endian-ness at compile-time."
#endif
#endif
            ;

#define BUFFER_SIZE 80

            static inline int isWhitespace(char c) {
                return c == ' ' || c == '\n' || c == '\t';
            }

// Reads a "word" from the fp and puts it into buffer and adds a null
// terminator.  i.e. it keeps reading until whitespace is reached.  Returns
// the number of characters read *not* including the whitespace, and
// returns -1 on an error.
            static int readWord(FILE *fp, char *buffer, int bufferLength) {
                int n;
                int c;

                if (bufferLength < 1) return -1;

                n = 0;
                c = fgetc(fp);
                while (c != EOF && !isWhitespace(c) && n < bufferLength) {
                    buffer[n] = c;
                    ++n;
                    c = fgetc(fp);
                }

                if (n < bufferLength) {
                    buffer[n] = '\0';
                    return n;
                }

                return -1;
            }

            static float *readImagePFM(const std::string &filename, int *xres, int *yres,
                                       int desire_channels, int *channels_in_file = nullptr) {
                float *data = nullptr;
                char buffer[BUFFER_SIZE];
                unsigned int nFloats;
                int nChannels, width, height;
                float scale;
                bool fileLittleEndian;

                FILE *fp = fopen(filename.c_str(), "rb");
                if (!fp) goto fail;

                // read either "Pf" or "PF"
                if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;

                if (strcmp(buffer, "Pf") == 0)
                    nChannels = 1;
                else if (strcmp(buffer, "PF") == 0)
                    nChannels = 3;
                else
                    goto fail;

                // read the rest of the header
                // read width
                if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
                width = atoi(buffer);
                *xres = width;

                // read height
                if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
                height = atoi(buffer);
                *yres = height;

                // read scale
                if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
                sscanf(buffer, "%f", &scale);

                // read the data
                nFloats = nChannels * width * height;
                data = new float[nFloats];
                // Flip in Y, as P*M has the origin at the lower left.
//                    for (int y = height - 1; y >= 0; --y) {
                for (int y = 0; y < height; y++) {
                    if (fread(&data[y * nChannels * width], sizeof(float),
                              nChannels * width, fp) != nChannels * width)
                        goto fail;
                }

                // apply endian conversian and scale if appropriate
                fileLittleEndian = (scale < 0.f);
                if (hostLittleEndian ^ fileLittleEndian) {
                    uint8_t bytes[4];
                    for (unsigned int i = 0; i < nFloats; ++i) {
                        memcpy(bytes, &data[i], 4);
                        std::swap(bytes[0], bytes[3]);
                        std::swap(bytes[1], bytes[2]);
                        memcpy(&data[i], bytes, 4);
                    }
                }
                if (std::abs(scale) != 1.f)
                    for (unsigned int i = 0; i < nFloats; ++i) data[i] *= std::abs(scale);

                // create RGBs...
                float *rgb = new float[width * height * desire_channels];
                if (nChannels == 1) {
                    for (int j = 0; j < height; j++) {
                        for (int i = 0; i < width; i++) {
                            int rgbOffset = (j * width) + i;
                            int imageOffset = (j * width + i) * desire_channels;
                            for (int ch = 0; ch < desire_channels; ch++) {
                                rgb[imageOffset + ch] = data[rgbOffset];
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < height; j++) {
                        for (int i = 0; i < width; i++) {
                            int rgbOffset = (j * width) + i;
                            Float frgb[3] = {data[3 * rgbOffset], data[3 * rgbOffset + 1], data[3 * rgbOffset + 2]};
                            int imageOffset = (j * width + i) * desire_channels;
                            for (int ch = 0; ch < desire_channels && ch < 3; ch++) {
                                rgb[imageOffset + ch] = frgb[ch];
                            }
                        }
                    }
                }

                delete[] data;
                fclose(fp);
                return rgb;

                fail:
                ASSERT(false, "Error reading PFM file \"%s\"" + filename);
                if (fp) fclose(fp);
                delete[] data;
                delete[] rgb;
                return nullptr;
            }

            static bool writeImagePFM(const std::string &filename, const Float *rgb,
                                      int width, int height) {
                FILE *fp;
                float scale;

                fp = fopen(filename.c_str(), "wb");
                if (!fp) {
                    ASSERT(false, "Unable to open output PFM file \"%s\"" + filename);
                    return false;
                }

                std::unique_ptr<float[]> scanline(new float[3 * width]);

                // only write 3 channel PFMs here...
                if (fprintf(fp, "PF\n") < 0) goto fail;

                // write the width and height, which must be positive
                if (fprintf(fp, "%d %d\n", width, height) < 0) goto fail;

                // write the scale, which encodes endianness
                scale = hostLittleEndian ? -1.f : 1.f;
                if (fprintf(fp, "%f\n", scale) < 0) goto fail;

                // write the data from bottom left to upper right as specified by
                // http://netpbm.sourceforge.net/doc/pfm.html
                // The raster is a sequence of pixels, packed one after another, with no
                // delimiters of any kind. They are grouped by row, with the pixels in each
                // row ordered left to right and the rows ordered bottom to top.
                for (int y = height - 1; y >= 0; y--) {
                    // in case Float is 'double', copy into a staging buffer that's
                    // definitely a 32-bit float...
                    for (int x = 0; x < 3 * width; ++x)
                        scanline[x] = rgb[y * width * 3 + x];
                    if (fwrite(&scanline[0], sizeof(float), width * 3, fp) <
                        (size_t) (width * 3))
                        goto fail;
                }

                fclose(fp);
                return true;

                fail:
                ASSERT(false, "Error writing PFM file \"%s\"" + filename);
                fclose(fp);
                return false;
            }
        }

        namespace png {
            static float *
            readImagePNG(const std::string &filename, int *width, int *height,
                         int desire_channels, int *channels_in_file = nullptr) {
                return stbReadImage(filename, width, height, desire_channels, channels_in_file);
            }
        }

        namespace jpg {
            static float *
            readImageJPG(const std::string &filename, int *width, int *height,
                         int desire_channels, int *channels_in_file = nullptr) {
                return stbReadImage(filename, width, height, desire_channels, channels_in_file);
            }
        }

        namespace tga {
            static float *
            readImageTGA(const std::string &filename, int *width, int *height,
                         int desire_channels, int *channels_in_file = nullptr) {
                return stbReadImage(filename, width, height, desire_channels, channels_in_file);
            }
        }

        float *readImage(const std::string &filename, int *width, int *height,
                         int desire_channels, int *channels_in_file) {
            // Check image type
            if (extensionEquals(filename, ".pfm")) {
                return pfm::readImagePFM(filename, width, height, desire_channels, channels_in_file);
            } else if (extensionEquals(filename, ".png")) {
                return png::readImagePNG(filename, width, height, desire_channels, channels_in_file);
            } else if (extensionEquals(filename, ".jpg") || extensionEquals(filename, ".jpeg")) {
                return jpg::readImageJPG(filename, width, height, desire_channels, channels_in_file);
            } else if (extensionEquals(filename, ".tga")) {
                return tga::readImageTGA(filename, width, height, desire_channels, channels_in_file);
            } else {
                ASSERT(false, "Unsupported image type: " + filename);
            }
        }

        void writeImage(const std::string filename, int width, int height, int channel,
                        const unsigned char *input_image) {
            unsigned char *image_buffer = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channel);
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int input_offset = (row * width + col) * channel;
                    int output_offset = ((height - row - 1) * width + col) * channel;
                    for (int ch = 0; ch < channel; ch++) {
                        image_buffer[output_offset + ch] = input_image[input_offset + ch];
                    }
                }
            }
            stbi_write_png(filename.c_str(), width, height, channel, image_buffer, 0);
        }
    }
}