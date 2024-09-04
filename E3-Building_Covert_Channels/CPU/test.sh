g++ sender.c -o sender.out -lpthread
g++ reciver.c -o reciver.out -lpthread
gcc error-rate.c -o error-rate.out
for i in {80..70..-5}; do
	echo $i  >> result.txt
	rm receive_signal
	./sender.out 1000 $i &
	(time ./reciver.out 1000 $i) 2>> result.txt
        ./error-rate.out 1000 >> result.txt
	sleep 8s
done
