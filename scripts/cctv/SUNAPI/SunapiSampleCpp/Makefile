CCOMP=g++
CFLAGS += -g -std=c++11
LDFLAGS= -lpthread -lz -lcurl


PROJECT = testClient
OBJECTS = main.o \
		restclient.o


INCLUDES = -I .

all: $(PROJECT)
.cpp.o:
	$(CCOMP) $(INCLUDES) -c $(CFLAGS) -o $@ $<

$(PROJECT): $(OBJECTS)
	$(CCOMP) $(OBJECTS) -o $(PROJECT) $(LDFLAGS)

clean:
	rm $(OBJECTS) $(PROJECT)
